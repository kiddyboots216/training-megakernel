"""Deterministic, resumable DCLM-to-WORLD8 shard production.

The GPU consumes fixed-geometry records through this example's workload store.
This module owns the earlier boundary: an ordered Parquet catalog; a cursor
precise enough to resume in the middle of one tokenized document; and atomic
production of shard sets.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import struct
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from training_megakernel.contract import VOCAB, WORLD_SIZE
from training_megakernel.shards import embedding_route

from .workload_store import (
    SHARDED_RANK_INDEX_SCHEMA,
    SHARDED_WORKLOAD_MANIFEST,
    SHARDED_WORKLOAD_SCHEMA,
    rank_index_name,
    rank_token_name,
)

DCLM_CATALOG_SCHEMA = "training_megakernel_dclm_catalog_v1"
DCLM_CURSOR_SCHEMA = "training_megakernel_dclm_cursor_v1"
# Every GPU's window: three documents and one padding segment of this many rows.
TEMPLATE_DOCUMENTS = 3
TEMPLATE_PAD_ROWS = 64
PARQUET_BATCH_ROWS = 128
NEXT_CURSOR_FILENAME = "next-cursor.json"
STREAM_CURSOR_FILENAME = "stream-cursors.jsonl"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True)
class DclmParquetFile:
    """One file in the exact catalog order."""

    path: str
    bytes: int
    row_group_rows: tuple[int, ...]


@dataclass(frozen=True)
class DclmCatalog:
    """The ordered DCLM files and their row groups."""

    dataset_repo: str
    dataset_revision: str
    files: tuple[DclmParquetFile, ...]

    @classmethod
    def load(cls, path: str | Path) -> DclmCatalog:
        value = json.loads(Path(path).read_bytes())
        if value.get("schema") != DCLM_CATALOG_SCHEMA:
            raise ValueError("unexpected DCLM catalog schema")
        return cls(
            dataset_repo=value["dataset_repo"],
            dataset_revision=value["dataset_revision"],
            files=tuple(
                DclmParquetFile(
                    path=row["path"],
                    bytes=row["bytes"],
                    row_group_rows=tuple(row["row_group_rows"]),
                )
                for row in value["files"]
            ),
        )

    def validate_coordinate(
        self,
        *,
        file_index: int,
        file_path: str,
        row_group_index: int,
        row_index: int,
    ) -> None:
        if not 0 <= file_index < len(self.files):
            raise ValueError("DCLM cursor file index is outside the catalog")
        source = self.files[file_index]
        if source.path != file_path:
            raise ValueError("DCLM cursor file path differs from its catalog index")
        if not 0 <= row_group_index < len(source.row_group_rows):
            raise ValueError("DCLM cursor row-group index is outside the file")
        if not 0 <= row_index < source.row_group_rows[row_group_index]:
            raise ValueError("DCLM cursor row index is outside the row group")


@dataclass(frozen=True)
class DclmTokenization:
    tokenizer_repo: str
    tokenizer_revision: str
    eod_token_id: int

    def __post_init__(self) -> None:
        if not 0 <= self.eod_token_id < VOCAB:
            raise ValueError(f"EOD token ID must be in [0, {VOCAB})")

    def manifest_data(self) -> dict[str, Any]:
        return {
            "add_special_tokens": False,
            "append_eod_token_id": self.eod_token_id,
            "tokenizer_repo": self.tokenizer_repo,
            "tokenizer_revision": self.tokenizer_revision,
        }


RecordId = str | int


@dataclass(frozen=True)
class DclmDocument:
    record_id: RecordId
    text: str


@dataclass(frozen=True)
class DclmCursor:
    """The next token to consume from the infinitely repeated catalog order."""

    dataset_repo: str
    dataset_revision: str
    tokenizer_repo: str
    tokenizer_revision: str
    eod_token_id: int
    epoch: int
    file_index: int
    file_path: str
    row_group_index: int
    row_index: int
    record_id: RecordId
    token_offset: int
    preprocessed_token_offset: int

    def validate(self, catalog: DclmCatalog, tokenization: DclmTokenization) -> None:
        if (
            self.dataset_repo != catalog.dataset_repo
            or self.dataset_revision != catalog.dataset_revision
        ):
            raise ValueError("DCLM cursor belongs to a different dataset revision")
        if (
            self.tokenizer_repo != tokenization.tokenizer_repo
            or self.tokenizer_revision != tokenization.tokenizer_revision
            or self.eod_token_id != tokenization.eod_token_id
        ):
            raise ValueError("DCLM cursor belongs to a different tokenization contract")
        catalog.validate_coordinate(
            file_index=self.file_index,
            file_path=self.file_path,
            row_group_index=self.row_group_index,
            row_index=self.row_index,
        )

    def checkpoint_data(self) -> dict[str, Any]:
        return {
            "dataset_repo": self.dataset_repo,
            "dataset_revision": self.dataset_revision,
            "eod_token_id": self.eod_token_id,
            "epoch": self.epoch,
            "file_index": self.file_index,
            "file_path": self.file_path,
            "preprocessed_token_offset": self.preprocessed_token_offset,
            "record_id": self.record_id,
            "row_group_index": self.row_group_index,
            "row_index": self.row_index,
            "schema": DCLM_CURSOR_SCHEMA,
            "token_offset": self.token_offset,
            "tokenizer_repo": self.tokenizer_repo,
            "tokenizer_revision": self.tokenizer_revision,
        }

    @classmethod
    def load(cls, path: str | Path) -> DclmCursor:
        value = json.loads(Path(path).read_bytes())
        if value.get("schema") != DCLM_CURSOR_SCHEMA:
            raise ValueError("unexpected DCLM cursor schema")
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class DclmStreamCursors:
    """The DCLM source cursor at every stream boundary of one shard set.

    ``at(i)`` is the source position of the first token of stream ``i``; the
    last boundary, ``stream_stop``, is the position after the final stream.
    """

    stream_start: int
    cursors: tuple[dict[str, Any], ...]

    @property
    def stream_stop(self) -> int:
        return self.stream_start + len(self.cursors) - 1

    def covers(self, stream_index: int) -> bool:
        return self.stream_start <= stream_index <= self.stream_stop

    def at(self, stream_index: int) -> dict[str, Any]:
        if not self.covers(stream_index):
            raise ValueError(
                f"stream {stream_index} is outside the shard set's stream boundaries "
                f"[{self.stream_start}, {self.stream_stop}]"
            )
        return copy.deepcopy(self.cursors[stream_index - self.stream_start])


def load_stream_cursors(root: str | Path, manifest: dict[str, Any]) -> DclmStreamCursors:
    """Read a shard set's stream-cursor sidecar once and index it by stream."""

    lines = (Path(root) / STREAM_CURSOR_FILENAME).read_bytes().splitlines()
    if len(lines) != manifest["stream_count"] + 1:
        raise ValueError("DCLM stream-cursor sidecar does not hold every stream boundary")
    return DclmStreamCursors(
        stream_start=manifest["stream_start"],
        cursors=tuple(json.loads(line)["cursor"] for line in lines),
    )


class DclmCorpusProducer:
    """Consume the catalog as one deterministic EOD-separated token stream."""

    def __init__(
        self,
        *,
        catalog: DclmCatalog,
        tokenization: DclmTokenization,
        reader: DclmParquetRowGroupReader,
        encode: Callable[[str], Sequence[int]],
        cursor: DclmCursor | None = None,
    ) -> None:
        self.catalog = catalog
        self.tokenization = tokenization
        self.reader = reader
        self.encode = encode
        self._iterator: Iterator[DclmDocument] | None = None
        self._iterator_key: tuple[int, int] | None = None
        self._iterator_next_row: int | None = None
        self._document_tokens: tuple[int, ...] | None = None

        if cursor is None:
            first = catalog.files[0]
            self._cursor = DclmCursor(
                dataset_repo=catalog.dataset_repo,
                dataset_revision=catalog.dataset_revision,
                tokenizer_repo=tokenization.tokenizer_repo,
                tokenizer_revision=tokenization.tokenizer_revision,
                eod_token_id=tokenization.eod_token_id,
                epoch=0,
                file_index=0,
                file_path=first.path,
                row_group_index=0,
                row_index=0,
                record_id="pending-initial-row",
                token_offset=0,
                preprocessed_token_offset=0,
            )
            document = self._read_current_document()
            self._cursor = replace(self._cursor, record_id=document.record_id)
        else:
            cursor.validate(catalog, tokenization)
            self._cursor = cursor
            document = self._read_current_document()
            if document.record_id != cursor.record_id:
                raise ValueError("DCLM cursor record ID differs from the source coordinate")
        self._document_tokens = self._tokens_for_document(document)
        if self._cursor.token_offset >= len(self._document_tokens):
            raise ValueError("DCLM cursor token offset is not normalized")

    @property
    def cursor(self) -> DclmCursor:
        return self._cursor

    def _read_current_document(self) -> DclmDocument:
        cursor = self._cursor
        key = (cursor.file_index, cursor.row_group_index)
        if (
            self._iterator is None
            or self._iterator_key != key
            or self._iterator_next_row != cursor.row_index
        ):
            source = self.catalog.files[cursor.file_index]
            self._iterator = iter(
                self.reader.iter_row_group(
                    source,
                    cursor.row_group_index,
                    start_row=cursor.row_index,
                )
            )
            self._iterator_key = key
            self._iterator_next_row = cursor.row_index
        try:
            document = next(self._iterator)
        except StopIteration as error:
            raise RuntimeError("DCLM reader ended before the catalog row-group extent") from error
        self._iterator_next_row = cursor.row_index + 1
        return document

    def _tokens_for_document(self, document: DclmDocument) -> tuple[int, ...]:
        return tuple(self.encode(document.text)) + (self.tokenization.eod_token_id,)

    def _advance_document(self) -> None:
        cursor = self._cursor
        source = self.catalog.files[cursor.file_index]
        file_index = cursor.file_index
        row_group_index = cursor.row_group_index
        row_index = cursor.row_index + 1
        epoch = cursor.epoch
        if row_index == source.row_group_rows[row_group_index]:
            row_index = 0
            row_group_index += 1
            if row_group_index == len(source.row_group_rows):
                row_group_index = 0
                file_index += 1
                if file_index == len(self.catalog.files):
                    file_index = 0
                    epoch += 1
        self._cursor = replace(
            cursor,
            epoch=epoch,
            file_index=file_index,
            file_path=self.catalog.files[file_index].path,
            row_group_index=row_group_index,
            row_index=row_index,
            token_offset=0,
        )
        document = self._read_current_document()
        self._cursor = replace(self._cursor, record_id=document.record_id)
        self._document_tokens = self._tokens_for_document(document)

    def take_tokens(self, count: int) -> tuple[int, ...]:
        result: list[int] = []
        remaining = count
        while remaining:
            tokens = self._document_tokens
            assert tokens is not None
            begin = self._cursor.token_offset
            take = min(len(tokens) - begin, remaining)
            result.extend(tokens[begin : begin + take])
            self._cursor = replace(
                self._cursor,
                token_offset=begin + take,
                preprocessed_token_offset=self._cursor.preprocessed_token_offset + take,
            )
            remaining -= take
            if self._cursor.token_offset == len(tokens):
                self._advance_document()
        return tuple(result)


class DclmParquetRowGroupReader:
    """Bounded-memory reader for the catalog's local Parquet files."""

    def __init__(self, base_directory: str | Path) -> None:
        self.base_directory = Path(base_directory)

    def iter_row_group(
        self,
        source: DclmParquetFile,
        row_group_index: int,
        *,
        start_row: int,
    ) -> Iterator[DclmDocument]:
        try:
            from pyarrow import parquet
        except ImportError as error:
            raise RuntimeError("DCLM Parquet input requires the 'dclm' extra") from error

        path = self.base_directory / source.path
        # The byte count and row-group extents catch a changed file cheaply.
        if path.stat().st_size != source.bytes:
            raise ValueError(f"DCLM Parquet byte count differs from the catalog: {path}")
        parquet_file = parquet.ParquetFile(path)
        metadata = parquet_file.metadata
        observed_groups = tuple(
            metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)
        )
        if observed_groups != source.row_group_rows:
            raise ValueError(f"DCLM Parquet row-group extents differ from the catalog: {path}")
        absolute_row = 0
        for batch in parquet_file.iter_batches(
            batch_size=PARQUET_BATCH_ROWS,
            row_groups=[row_group_index],
            columns=["id", "text"],
            use_threads=False,
        ):
            ids = batch.column(batch.schema.get_field_index("id")).to_pylist()
            texts = batch.column(batch.schema.get_field_index("text")).to_pylist()
            for record_id, text in zip(ids, texts, strict=True):
                if absolute_row >= start_row:
                    yield DclmDocument(record_id=record_id, text=text)
                absolute_row += 1


def load_tokenizer_encoder(path: str | Path) -> Callable[[str], Sequence[int]]:
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError("DCLM tokenization requires the 'dclm' extra") from error
    tokenizer = Tokenizer.from_file(str(path))
    if tokenizer.get_vocab_size(with_added_tokens=True) > VOCAB:
        raise ValueError("tokenizer vocabulary exceeds the compiled vocabulary")

    def encode(text: str) -> Sequence[int]:
        return tokenizer.encode(text, add_special_tokens=False).ids

    return encode


def packing_template(sequence: int) -> tuple[tuple[int, ...], int]:
    """Every GPU's documents and padding at S tokens per GPU.

    Three documents of (S - 64) / 3 tokens (the first ones one longer when that
    does not divide) and one 64-row padding segment, which starts on a 64-row
    boundary since S is a multiple of 1,024.
    """

    base, remainder = divmod(sequence - TEMPLATE_PAD_ROWS, TEMPLATE_DOCUMENTS)
    documents = tuple(base + (index < remainder) for index in range(TEMPLATE_DOCUMENTS))
    return documents, TEMPLATE_PAD_ROWS


def publish_dclm_shard_batch(
    producer: DclmCorpusProducer,
    output: str | Path,
    *,
    sequence: int,
    steps: int,
    stream_start: int,
    embedding_route_records: int,
) -> dict[str, Any]:
    """Publish one shard set and its next cursor.

    ``sequence`` and ``embedding_route_records`` are the tokens per GPU and the
    route capacity of the bundle that will train on the shard set; every
    window must fit the capacity.
    """

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    documents, pad = packing_template(sequence)
    tokens_per_step = WORLD_SIZE * sum(documents)
    expected_offset = stream_start * tokens_per_step
    if producer.cursor.preprocessed_token_offset != expected_offset:
        raise ValueError(
            "DCLM cursor token offset does not match the global stream index: "
            f"{producer.cursor.preprocessed_token_offset} != {expected_offset}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    cursor_start = producer.cursor
    window = {"docs": list(documents), "pad": pad}
    stream_boundaries = [cursor_start.checkpoint_data()]
    try:
        with ExitStack() as stack:
            token_files = [
                stack.enter_context((staging / rank_token_name(rank)).open("xb"))
                for rank in range(WORLD_SIZE)
            ]
            for _step in range(steps):
                for token_file in token_files:
                    record = producer.take_tokens(sum(documents)) + (0,) * pad
                    embedding_route(
                        torch.tensor(record, dtype=torch.int32),
                        sequence=sequence,
                        embedding_route_records=embedding_route_records,
                    )
                    token_file.write(struct.pack(f"<{sequence}i", *record))
                stream_boundaries.append(producer.cursor.checkpoint_data())

        for rank in range(WORLD_SIZE):
            index = {
                "rank": rank,
                "schema": SHARDED_RANK_INDEX_SCHEMA,
                "stream_count": steps,
                "stream_start": stream_start,
                "token_file": rank_token_name(rank),
                "windows": [
                    {**window, "stream_index": stream_start + step} for step in range(steps)
                ],
            }
            (staging / rank_index_name(rank)).write_bytes(_canonical_json_bytes(index))
        (staging / STREAM_CURSOR_FILENAME).write_bytes(
            b"".join(
                _canonical_json_bytes(
                    {"cursor": cursor, "stream_index": stream_start + offset}
                )
                + b"\n"
                for offset, cursor in enumerate(stream_boundaries)
            )
        )
        cursor_stop = producer.cursor
        (staging / NEXT_CURSOR_FILENAME).write_bytes(
            _canonical_json_bytes(cursor_stop.checkpoint_data())
        )
        manifest = {
            "embedding_route_records_per_source_owner": embedding_route_records,
            "schema": SHARDED_WORKLOAD_SCHEMA,
            "sequence": sequence,
            "source": {
                "cursor_start": cursor_start.checkpoint_data(),
                "cursor_stop": cursor_stop.checkpoint_data(),
                "dataset_repo": producer.catalog.dataset_repo,
                "dataset_revision": producer.catalog.dataset_revision,
                "tokenization": producer.tokenization.manifest_data(),
            },
            "stream_count": steps,
            "stream_start": stream_start,
        }
        (staging / SHARDED_WORKLOAD_MANIFEST).write_bytes(_canonical_json_bytes(manifest))
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "cursor_stop": cursor_stop.checkpoint_data(),
        "next_cursor": str(destination / NEXT_CURSOR_FILENAME),
        "output": str(destination),
        "preprocessed_tokens": steps * tokens_per_step,
        "stream_count": steps,
        "stream_start": stream_start,
    }


__all__ = (
    "DCLM_CATALOG_SCHEMA",
    "DCLM_CURSOR_SCHEMA",
    "DclmCatalog",
    "DclmCorpusProducer",
    "DclmCursor",
    "DclmDocument",
    "DclmParquetFile",
    "DclmParquetRowGroupReader",
    "DclmStreamCursors",
    "DclmTokenization",
    "load_stream_cursors",
    "load_tokenizer_encoder",
    "packing_template",
    "publish_dclm_shard_batch",
)
