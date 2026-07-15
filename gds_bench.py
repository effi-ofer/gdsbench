#!/usr/bin/env python3
"""GDS benchmark harness"""

import argparse
import ctypes
import os
import threading
import time

import torch

# --- cuFile ctypes bindings ---

CU_FILE_SUCCESS = 0
CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1

libcufile = ctypes.CDLL("libcufile.so", mode=ctypes.RTLD_GLOBAL)

class CUfileError_t(ctypes.Structure):
    _fields_ = [("err", ctypes.c_int), ("cu_err", ctypes.c_int)]

CUfileHandle_t = ctypes.c_void_p

class _HandleUnion(ctypes.Union):
    _fields_ = [("fd", ctypes.c_int), ("handle", ctypes.c_void_p)]

class CUfileDescr_t(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint),
        ("handle", _HandleUnion),
        ("fs_ops", ctypes.c_void_p),
    ]

libcufile.cuFileDriverOpen.restype = CUfileError_t
libcufile.cuFileDriverOpen.argtypes = []

libcufile.cuFileDriverClose.restype = CUfileError_t
libcufile.cuFileDriverClose.argtypes = []

libcufile.cuFileHandleRegister.restype = CUfileError_t
libcufile.cuFileHandleRegister.argtypes = [
    ctypes.POINTER(CUfileHandle_t),
    ctypes.POINTER(CUfileDescr_t),
]

libcufile.cuFileHandleDeregister.restype = None
libcufile.cuFileHandleDeregister.argtypes = [CUfileHandle_t]

libcufile.cuFileBufRegister.restype = CUfileError_t
libcufile.cuFileBufRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

libcufile.cuFileBufDeregister.restype = CUfileError_t
libcufile.cuFileBufDeregister.argtypes = [ctypes.c_void_p]

libcufile.cuFileStreamRegister.restype = CUfileError_t
libcufile.cuFileStreamRegister.argtypes = [ctypes.c_void_p, ctypes.c_uint]

libcufile.cuFileStreamDeregister.restype = CUfileError_t
libcufile.cuFileStreamDeregister.argtypes = [ctypes.c_void_p]

libcufile.cuFileWriteAsync.restype = CUfileError_t
libcufile.cuFileWriteAsync.argtypes = [
    CUfileHandle_t,                      # handle
    ctypes.c_void_p,                     # devPtr
    ctypes.POINTER(ctypes.c_size_t),     # size_p
    ctypes.POINTER(ctypes.c_ssize_t),    # file_offset_p
    ctypes.POINTER(ctypes.c_ssize_t),    # buf_offset_p
    ctypes.POINTER(ctypes.c_ssize_t),    # bytes_written_p
    ctypes.c_void_p,                     # stream
]

libcufile.cuFileReadAsync.restype = CUfileError_t
libcufile.cuFileReadAsync.argtypes = [
    CUfileHandle_t,                      # handle
    ctypes.c_void_p,                     # devPtr
    ctypes.POINTER(ctypes.c_size_t),     # size_p
    ctypes.POINTER(ctypes.c_ssize_t),    # file_offset_p
    ctypes.POINTER(ctypes.c_ssize_t),    # buf_offset_p
    ctypes.POINTER(ctypes.c_ssize_t),    # bytes_read_p
    ctypes.c_void_p,                     # stream
]

libcufile.cuFileRead.restype = ctypes.c_ssize_t
libcufile.cuFileRead.argtypes = [
    CUfileHandle_t,     # handle
    ctypes.c_void_p,    # devPtr
    ctypes.c_size_t,    # size
    ctypes.c_long,      # file_offset (off_t)
    ctypes.c_long,      # buf_offset (off_t)
]

libcufile.cuFileWrite.restype = ctypes.c_ssize_t
libcufile.cuFileWrite.argtypes = [
    CUfileHandle_t,     # handle
    ctypes.c_void_p,    # devPtr
    ctypes.c_size_t,    # size
    ctypes.c_long,      # file_offset (off_t)
    ctypes.c_long,      # buf_offset (off_t)
]

# --- Backends ---

def _register_handle(fd):
    descr = CUfileDescr_t()
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD
    descr.handle.fd = fd
    descr.fs_ops = None
    handle = CUfileHandle_t()
    err = libcufile.cuFileHandleRegister(ctypes.byref(handle), ctypes.byref(descr))
    assert err.err == CU_FILE_SUCCESS, f"cuFileHandleRegister failed: {err.err}"
    return handle.value


class SyncBackend:
    name = "cuFileRead / cuFileWrite (synchronous)"

    def open_file(self, filepath, flags, mode=0o644):
        fd = os.open(filepath, flags, mode)
        handle = _register_handle(fd)
        return {"fd": fd, "handle": handle}

    def write_all(self, ctxs, buf_ptr, size):
        def _write(ctx):
            torch.cuda.set_device(0)
            ret = libcufile.cuFileWrite(ctx["handle"], buf_ptr, size, 0, 0)
            assert ret >= 0, f"cuFileWrite failed with error: {ret}"
            assert ret == size, f"cuFileWrite short write: {ret}/{size}"
        threads = [threading.Thread(target=_write, args=(ctx,)) for ctx in ctxs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def read_all(self, ctxs, buf_ptr, size):
        def _read(ctx):
            torch.cuda.set_device(0)
            ret = libcufile.cuFileRead(ctx["handle"], buf_ptr, size, 0, 0)
            assert ret >= 0, f"cuFileRead failed with error: {ret}"
            assert ret == size, f"cuFileRead short read: {ret}/{size}"
        threads = [threading.Thread(target=_read, args=(ctx,)) for ctx in ctxs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def close(self, ctx):
        libcufile.cuFileHandleDeregister(ctx["handle"])
        os.close(ctx["fd"])


class AsyncBackend:
    name = "cuFileReadAsync / cuFileWriteAsync"

    def open_file(self, filepath, flags, mode=0o644):
        fd = os.open(filepath, flags, mode)
        handle = _register_handle(fd)
        stream = torch.cuda.Stream()
        err = libcufile.cuFileStreamRegister(stream.cuda_stream, 0)
        assert err.err == CU_FILE_SUCCESS, f"cuFileStreamRegister failed: {err.err}"
        return {"fd": fd, "handle": handle, "stream": stream}

    def write_all(self, ctxs, buf_ptr, size):
        for ctx in ctxs:
            size_val = ctypes.c_size_t(size)
            file_offset = ctypes.c_ssize_t(0)
            buf_offset = ctypes.c_ssize_t(0)
            bytes_done = ctypes.c_ssize_t(0)
            ctx["_params"] = (size_val, file_offset, buf_offset, bytes_done)
            err = libcufile.cuFileWriteAsync(
                ctx["handle"], buf_ptr,
                ctypes.byref(size_val),
                ctypes.byref(file_offset),
                ctypes.byref(buf_offset),
                ctypes.byref(bytes_done),
                ctx["stream"].cuda_stream,
            )
            assert err.err == CU_FILE_SUCCESS, f"cuFileWriteAsync failed: {err.err}"
        for ctx in ctxs:
            ctx["stream"].synchronize()

    def read_all(self, ctxs, buf_ptr, size):
        for ctx in ctxs:
            size_val = ctypes.c_size_t(size)
            file_offset = ctypes.c_ssize_t(0)
            buf_offset = ctypes.c_ssize_t(0)
            bytes_done = ctypes.c_ssize_t(0)
            ctx["_params"] = (size_val, file_offset, buf_offset, bytes_done)
            err = libcufile.cuFileReadAsync(
                ctx["handle"], buf_ptr,
                ctypes.byref(size_val),
                ctypes.byref(file_offset),
                ctypes.byref(buf_offset),
                ctypes.byref(bytes_done),
                ctx["stream"].cuda_stream,
            )
            assert err.err == CU_FILE_SUCCESS, f"cuFileReadAsync failed: {err.err}"
        for ctx in ctxs:
            ctx["stream"].synchronize()

    def close(self, ctx):
        libcufile.cuFileStreamDeregister(ctx["stream"].cuda_stream)
        libcufile.cuFileHandleDeregister(ctx["handle"])
        os.close(ctx["fd"])


BACKENDS = {"async": AsyncBackend, "cufile": SyncBackend}

# --- Benchmark harness ---

def parse_args():
    parser = argparse.ArgumentParser(description="GDS API benchmark harness")
    parser.add_argument("--api", default="async", choices=list(BACKENDS.keys()),
                        help="cuFile API to use (default: async)")
    parser.add_argument("--tokens", type=int, default=1024,
                        help="Number of tokens per I/O operation")
    parser.add_argument("--writes", type=int, default=1,
                        help="Number of write operations to issue")
    parser.add_argument("--reads", type=int, default=1,
                        help="Number of read operations to issue")
    parser.add_argument("--dir", default=".",
                        help="Directory for test files (must support O_DIRECT)")
    return parser.parse_args()


def run_benchmark(args, backend):
    single_token_kv_size = 131072
    io_size = args.tokens * single_token_kv_size
    aligned_size = ((io_size + 4095) // 4096) * 4096

    print(f"API:        {backend.name}")
    print(f"Token size: {io_size} bytes (131072 bytes/token x {args.tokens} tokens)")
    print(f"Aligned IO: {aligned_size} bytes")
    print(f"Writes:     {args.writes}")
    print(f"Reads:      {args.reads}")
    print(f"Directory:  {args.dir}")
    print()

    err = libcufile.cuFileDriverOpen()
    assert err.err == CU_FILE_SUCCESS, f"cuFileDriverOpen failed: {err.err}"

    write_buf = torch.full((aligned_size,), 0xAB, dtype=torch.uint8, device="cuda")
    buf_ptr = write_buf.data_ptr()
    err = libcufile.cuFileBufRegister(buf_ptr, aligned_size, 0)
    assert err.err == CU_FILE_SUCCESS, f"cuFileBufRegister write failed: {err.err}"

    # --- Writes ---
    write_files = [os.path.join(args.dir, f"gds_bench_{i}.bin") for i in range(args.writes)]
    write_ctxs = [backend.open_file(f, os.O_CREAT | os.O_WRONLY | os.O_DIRECT) for f in write_files]

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    backend.write_all(write_ctxs, buf_ptr, aligned_size)

    t1 = time.perf_counter()

    for ctx in write_ctxs:
        backend.close(ctx)

    elapsed_ms = (t1 - t0) * 1000
    total_bytes = aligned_size * args.writes
    throughput_gibs = (total_bytes / (1 << 30)) / (t1 - t0) if t1 > t0 else 0
    print(f"Writes complete: {args.writes} ops, {elapsed_ms:.2f} ms, {throughput_gibs:.3f} GiB/s")

    # --- Reads ---
    read_buf = torch.zeros(aligned_size, dtype=torch.uint8, device="cuda")
    read_ptr = read_buf.data_ptr()
    err = libcufile.cuFileBufRegister(read_ptr, aligned_size, 0)
    assert err.err == CU_FILE_SUCCESS, f"cuFileBufRegister read failed: {err.err}"

    read_files = [os.path.join(args.dir, f"gds_bench_{i}.bin") for i in range(args.reads)]
    read_ctxs = [backend.open_file(f, os.O_RDONLY | os.O_DIRECT) for f in read_files]

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    backend.read_all(read_ctxs, read_ptr, aligned_size)

    t1 = time.perf_counter()

    for ctx in read_ctxs:
        backend.close(ctx)

    elapsed_ms = (t1 - t0) * 1000
    total_bytes = aligned_size * args.reads
    throughput_gibs = (total_bytes / (1 << 30)) / (t1 - t0) if t1 > t0 else 0
    print(f"Reads complete:  {args.reads} ops, {elapsed_ms:.2f} ms, {throughput_gibs:.3f} GiB/s")

    # Verify
    if torch.all(read_buf == 0xAB).item():
        print("\nVerification: PASS")
    else:
        print("\nVerification: FAIL")

    # Cleanup
    err = libcufile.cuFileBufDeregister(read_ptr)
    assert err.err == CU_FILE_SUCCESS, f"cuFileBufDeregister read failed: {err.err}"
    err = libcufile.cuFileBufDeregister(buf_ptr)
    assert err.err == CU_FILE_SUCCESS, f"cuFileBufDeregister write failed: {err.err}"
    libcufile.cuFileDriverClose()
    for filepath in write_files:
        os.unlink(filepath)
    print("Done. Test files removed.")


def main():
    args = parse_args()
    print(f"PyTorch:    {torch.__version__}")
    print(f"CUDA:       {torch.version.cuda}")
    print(f"Device:     {torch.cuda.get_device_name(0)}")
    run_benchmark(args, BACKENDS[args.api]())


if __name__ == "__main__":
    main()
