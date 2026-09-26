"""CUDA runtime calls that map a host page range into a GPU's address space."""

from __future__ import annotations

import ctypes
from functools import cache

_HOST_REGISTER_PORTABLE_MAPPED = 0x01 | 0x02


@cache
def cudart() -> ctypes.CDLL:
    library = ctypes.CDLL("libcudart.so.13")
    library.cudaSetDevice.argtypes = (ctypes.c_int,)
    library.cudaHostRegister.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint)
    library.cudaHostGetDevicePointer.argtypes = (
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    )
    library.cudaHostUnregister.argtypes = (ctypes.c_void_p,)
    library.cudaDeviceSetLimit.argtypes = (ctypes.c_int, ctypes.c_size_t)
    return library


def _check(status: int, operation: str) -> None:
    if status:
        raise RuntimeError(f"{operation} failed with cudaError={status}")


def register_mapped(address: int, size: int, device_index: int) -> int:
    """Register ``size`` bytes at ``address`` as mapped host memory; return the device alias."""

    library = cudart()
    _check(library.cudaSetDevice(device_index), "cudaSetDevice")
    _check(
        library.cudaHostRegister(address, size, _HOST_REGISTER_PORTABLE_MAPPED),
        "cudaHostRegister",
    )
    device_pointer = ctypes.c_void_p()
    status = library.cudaHostGetDevicePointer(ctypes.byref(device_pointer), address, 0)
    if status or not device_pointer.value:
        library.cudaHostUnregister(address)
        raise RuntimeError(f"cudaHostGetDevicePointer failed with cudaError={status}")
    return int(device_pointer.value)


def unregister_mapped(address: int) -> None:
    _check(cudart().cudaHostUnregister(address), "cudaHostUnregister")
