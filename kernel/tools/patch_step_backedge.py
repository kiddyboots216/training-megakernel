#!/usr/bin/env python3
"""Turn the step loop's predicated EXIT into a branch to the kernel's first instruction.

The program ends each step with a branch that skips the exit sentinel's store when another step
follows (kernel/program/resident_step.py), and ptxas assembles that branch as a
predicated EXIT just before the store. ptxas drops the register-split instructions when the
step sits inside a source-level loop, so the loop is made here, after assembly:
``patch_backedge`` rewrites that one 128-bit instruction into a BRA with the same predicate,
changing its opcode, its relative target and the target's sign-extension bits in the control
word, and keeping the rest of its scheduling control. The same 16 bytes are rewritten in the
cubin, in the fatbin that embeds it and in the object file that embeds the fatbin; no other byte
may change. ``kernel/build.py`` calls ``patch_backedge``.
"""

from __future__ import annotations

import os
import re
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

SECTION = re.compile(
    r"\[\s*\d+\]\s+(\.text\.kernel_\S+)\s+PROGBITS\s+"
    r"[0-9a-fA-F]+\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)"
)
INSTRUCTION = re.compile(r"^\s*/\*([0-9a-fA-F]+)\*/\s+(.*?)\s*;\s*$")
REGISTER = r"R(?:Z|[0-9]+)"
# ptxas materializes the sentinel and the zero high word with MOV or IMAD.MOV.U32.
SENTINEL = re.compile(
    rf"^(?:MOV (?P<mov>{REGISTER}), 0x76543210|IMAD\.MOV\.U32 (?P<imad>{REGISTER}), RZ, RZ, 0x76543210)$"
)
LDC64 = re.compile(rf"^LDC\.64 ({REGISTER}), .+$")
ZERO_MOV = re.compile(
    rf"^(?:MOV {REGISTER}, (?:0x0|RZ)|IMAD\.MOV\.U32 {REGISTER}, RZ, RZ, 0x0)$"
)
PREDICATED_EXIT = re.compile(r"^@(!?)P([0-6]) EXIT$")
EXIT_OPCODE = 0x94D


def predicated_exit_low16(instruction: str) -> int | None:
    """Low 16 bits of ``@[!]Pn EXIT``: the EXIT opcode with the predicate in bits 12-15."""

    match = PREDICATED_EXIT.fullmatch(instruction)
    if match is None:
        return None
    negated, register = match.group(1) == "!", int(match.group(2))
    return ((int(negated) << 3 | register) << 12) | EXIT_OPCODE


@dataclass(frozen=True)
class TailSite:
    """The step's matched six-instruction tail.

    ``tail_pc`` is the predicated EXIT's PC and ``expected_low16`` its low 16 bits.
    """

    tail_pc: int
    expected_low16: int


def kernel_section(cubin: Path) -> tuple[int, int]:
    """The file offset and size of the cubin's one kernel text section."""

    output = subprocess.check_output(
        ["readelf", "-SW", str(cubin)], text=True, stderr=subprocess.DEVNULL
    )
    matches = SECTION.findall(output)
    if len(matches) != 1:
        raise RuntimeError(f"expected one kernel text section, found {matches!r}")
    _name, offset, size = matches[0]
    return int(offset, 16), int(size, 16)


def unique_embedded_offset(container: bytes | bytearray, payload: bytes | bytearray) -> int:
    """Where ``payload`` sits in ``container``.

    Its first 64 bytes must occur only once, and the whole payload must match there.
    """

    needle = bytes(payload[:64])
    first = bytes(container).find(needle)
    if first < 0 or bytes(container).find(needle, first + 1) >= 0:
        raise RuntimeError("payload prefix was not embedded exactly once")
    if bytes(container[first : first + len(payload)]) != bytes(payload):
        raise RuntimeError("embedded payload is not byte-identical")
    return first


def parse_instructions(disassembly: str) -> list[tuple[int, str]]:
    """The (PC, instruction) pairs of an nvdisasm listing, in order, whitespace collapsed."""

    result: list[tuple[int, str]] = []
    for line in disassembly.splitlines():
        match = INSTRUCTION.match(line)
        if match is not None:
            result.append((int(match.group(1), 16), " ".join(match.group(2).split())))
    if not result:
        raise RuntimeError("nvdisasm emitted no parseable instructions")
    return result


def discover_tail_site(disassembly: str) -> TailSite:
    """Find the step's predicated EXIT by the six consecutive instructions around the sentinel.

    The tail must read ``@[!]Pn EXIT``, an ``LDC.64`` of the store's base address, the
    ``0x76543210`` sentinel and zero moves, the ``STG.E.64`` of the sentinel through that base,
    and the final ``EXIT``. Requiring the whole sequence keeps the kernel's other predicated
    EXITs from matching; a single match is required.
    """

    instructions = parse_instructions(disassembly)
    candidates: list[TailSite] = []
    for index, (_pc, sentinel_body) in enumerate(instructions):
        sentinel_match = SENTINEL.fullmatch(sentinel_body)
        if sentinel_match is None or index < 2 or index + 3 >= len(instructions):
            continue
        window = instructions[index - 2 : index + 4]
        pcs = [pc for pc, _ in window]
        if pcs != list(range(pcs[0], pcs[0] + 6 * 16, 16)):
            continue
        bodies = [body for _, body in window]
        expected_low16 = predicated_exit_low16(bodies[0])
        ldc = LDC64.fullmatch(bodies[1])
        zero = ZERO_MOV.fullmatch(bodies[3])
        sentinel_register = sentinel_match.group("mov") or sentinel_match.group("imad")
        if (
            expected_low16 is None
            or ldc is None
            or zero is None
            or not bodies[4].startswith("STG.E.64 ")
            or not bodies[4].endswith(f", {sentinel_register}")
            or f"[{ldc.group(1)}.64" not in bodies[4]
            or bodies[5] != "EXIT"
        ):
            continue
        candidates.append(TailSite(tail_pc=window[0][0], expected_low16=expected_low16))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one step tail ending in the exit sentinel, found {len(candidates)}"
        )
    return candidates[0]


def disassemble(cubin: Path) -> str:
    completed = subprocess.run(
        ["nvdisasm", str(cubin)],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"nvdisasm failed for {cubin} ({completed.returncode}):\n{completed.stderr}"
        )
    return completed.stdout


def encoded_bra(old_low: int, old_control: int, *, pc: int) -> tuple[int, int]:
    """The low and control words of a BRA to the kernel's start, replacing the EXIT at ``pc``."""

    relative_quarters = (-pc - 16) // 4
    low_eight = relative_quarters & 0xFF
    high_signed = (relative_quarters - low_eight) // 64
    if not -(1 << 31) <= high_signed < (1 << 31):
        raise ValueError("BRA target is outside the encoded range")
    # ptxas chooses which predicate register holds the "another step" flag, so the predicate
    # bits are carried over from the EXIT.
    predicate = old_low & 0xFF00
    new_low = (
        ((high_signed & 0xFFFFFFFF) << 32)
        | (low_eight << 16)
        | predicate
        | 0x47
    )
    # Hopper keeps the target's sign extension in the control word's low 18 bits, all ones for
    # this backward branch; the other control bits are kept.
    new_control = old_control | 0x3FFFF
    return new_low, new_control


def write_exact(path: Path, data: bytearray) -> None:
    """Write ``data`` to ``path``, making the file writable first."""

    os.chmod(path, path.stat().st_mode | 0o200)
    path.write_bytes(data)


def patch_backedge(directory: Path) -> None:
    """Patch ``image.cubin``, ``image.fatbin`` and ``image.o`` in ``directory`` in place.

    The step's EXIT becomes a branch to the kernel's first instruction.
    """

    directory = directory.resolve()
    cubin_path = directory / "image.cubin"
    fatbin_path = directory / "image.fatbin"
    object_path = directory / "image.o"
    discovered = discover_tail_site(disassemble(cubin_path))
    tail_pc = discovered.tail_pc
    expected_low16 = discovered.expected_low16
    cubin = bytearray(cubin_path.read_bytes())
    fatbin = bytearray(fatbin_path.read_bytes())
    image_object = bytearray(object_path.read_bytes())

    section_offset, section_size = kernel_section(cubin_path)
    if not 0 <= tail_pc <= section_size - 16:
        raise ValueError("tail PC is outside the kernel text section")
    cubin_instruction_offset = section_offset + tail_pc
    old_low, old_control = struct.unpack_from("<QQ", cubin, cubin_instruction_offset)
    if old_low != expected_low16:
        raise ValueError(
            f"terminal EXIT carries unexpected operand bits: 0x{old_low:016x}"
        )
    new_low, new_control = encoded_bra(old_low, old_control, pc=tail_pc)

    fatbin_cubin_offset = unique_embedded_offset(fatbin, cubin)
    object_fatbin_offset = unique_embedded_offset(image_object, fatbin)
    new_instruction = struct.pack("<QQ", new_low, new_control)
    cubin[cubin_instruction_offset : cubin_instruction_offset + 16] = new_instruction
    fatbin_offset = fatbin_cubin_offset + cubin_instruction_offset
    fatbin[fatbin_offset : fatbin_offset + 16] = new_instruction
    object_offset = object_fatbin_offset + fatbin_offset
    image_object[object_offset : object_offset + 16] = new_instruction

    write_exact(cubin_path, cubin)
    write_exact(fatbin_path, fatbin)
    write_exact(object_path, image_object)
