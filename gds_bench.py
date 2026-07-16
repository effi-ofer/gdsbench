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

# --- cuFile Batch IO structures ---

CUFILE_READ = 0
CUFILE_WRITE = 1

class _BatchParams(ctypes.Structure):
    _fields_ = [
        ("devPtr_base", ctypes.c_void_p),
        ("file_offset", ctypes.c_long),   # off_t
        ("devPtr_offset", ctypes.c_long),  # off_t
        ("size", ctypes.c_size_t),
    ]

class _BatchUnion(ctypes.Union):
    _fields_ = [("batch", _BatchParams)]

class CUfileIOParams_t(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_uint),       # CUfileBatchMode_t
        ("u", _BatchUnion),
        ("fh", CUfileHandle_t),
        ("opcode", ctypes.c_uint),     # CUfileOpcode_t
        ("cookie", ctypes.c_void_p),
    ]

class CUfileIOEvents_t(ctypes.Structure):
    _fields_ = [
        ("cookie", ctypes.c_void_p),
        ("status", ctypes.c_uint),     # CUfileStatus_t
        ("ret", ctypes.c_size_t),
    ]

CUfileBatchHandle_t = ctypes.c_void_p

class Timespec(ctypes.Structure):
    _fields_ = [
        ("tv_sec", ctypes.c_long),
        ("tv_nsec", ctypes.c_long),
    ]

libcufile.cuFileBatchIOSetUp.restype = CUfileError_t
libcufile.cuFileBatchIOSetUp.argtypes = [
    ctypes.POINTER(CUfileBatchHandle_t),
    ctypes.c_uint,
]

libcufile.cuFileBatchIOSubmit.restype = CUfileError_t
libcufile.cuFileBatchIOSubmit.argtypes = [
    CUfileBatchHandle_t,
    ctypes.c_uint,
    ctypes.POINTER(CUfileIOParams_t),
    ctypes.c_uint,
]

libcufile.cuFileBatchIOGetStatus.restype = CUfileError_t
libcufile.cuFileBatchIOGetStatus.argtypes = [
    CUfileBatchHandle_t,
    ctypes.c_uint,                        # min_nr
    ctypes.POINTER(ctypes.c_uint),        # nr (out)
    ctypes.POINTER(CUfileIOEvents_t),
    ctypes.POINTER(Timespec),
]

libcufile.cuFileBatchIODestroy.restype = None
libcufile.cuFileBatchIODestroy.argtypes = [CUfileBatchHandle_t]

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

    def write_all(self, ctxs, size):
        def _write(ctx):
            torch.cuda.set_device(0)
            ret = libcufile.cuFileWrite(ctx["handle"], ctx["buf_ptr"], size, 0, 0)
            assert ret >= 0, f"cuFileWrite failed with error: {ret}"
            assert ret == size, f"cuFileWrite short write: {ret}/{size}"
        threads = [threading.Thread(target=_write, args=(ctx,)) for ctx in ctxs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def read_all(self, ctxs, size):
        def _read(ctx):
            torch.cuda.set_device(0)
            ret = libcufile.cuFileRead(ctx["handle"], ctx["buf_ptr"], size, 0, 0)
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

    NUM_STREAMS = 64

    def __init__(self):
        self._streams = []
        for _ in range(self.NUM_STREAMS):
            s = torch.cuda.Stream()
            err = libcufile.cuFileStreamRegister(s.cuda_stream, 0)
            assert err.err == CU_FILE_SUCCESS, f"cuFileStreamRegister failed: {err.err}"
            self._streams.append(s)
        self._next = 0

    def _get_stream(self):
        s = self._streams[self._next % self.NUM_STREAMS]
        self._next += 1
        return s

    def open_file(self, filepath, flags, mode=0o644):
        fd = os.open(filepath, flags, mode)
        handle = _register_handle(fd)
        return {"fd": fd, "handle": handle}

    def write_all(self, ctxs, size):
        self._next = 0
        for ctx in ctxs:
            stream = self._get_stream()
            size_val = ctypes.c_size_t(size)
            file_offset = ctypes.c_ssize_t(0)
            buf_offset = ctypes.c_ssize_t(0)
            bytes_done = ctypes.c_ssize_t(0)
            ctx["_params"] = (size_val, file_offset, buf_offset, bytes_done)
            err = libcufile.cuFileWriteAsync(
                ctx["handle"], ctx["buf_ptr"],
                ctypes.byref(size_val),
                ctypes.byref(file_offset),
                ctypes.byref(buf_offset),
                ctypes.byref(bytes_done),
                stream.cuda_stream,
            )
            assert err.err == CU_FILE_SUCCESS, f"cuFileWriteAsync failed: {err.err}"
        for s in self._streams:
            s.synchronize()

    def read_all(self, ctxs, size):
        self._next = 0
        for ctx in ctxs:
            stream = self._get_stream()
            size_val = ctypes.c_size_t(size)
            file_offset = ctypes.c_ssize_t(0)
            buf_offset = ctypes.c_ssize_t(0)
            bytes_done = ctypes.c_ssize_t(0)
            ctx["_params"] = (size_val, file_offset, buf_offset, bytes_done)
            err = libcufile.cuFileReadAsync(
                ctx["handle"], ctx["buf_ptr"],
                ctypes.byref(size_val),
                ctypes.byref(file_offset),
                ctypes.byref(buf_offset),
                ctypes.byref(bytes_done),
                stream.cuda_stream,
            )
            assert err.err == CU_FILE_SUCCESS, f"cuFileReadAsync failed: {err.err}"
        for s in self._streams:
            s.synchronize()

    def close(self, ctx):
        libcufile.cuFileHandleDeregister(ctx["handle"])
        os.close(ctx["fd"])


class BatchBackend:
    name = "cuFileBatchIO"

    def open_file(self, filepath, flags, mode=0o644):
        fd = os.open(filepath, flags, mode)
        handle = _register_handle(fd)
        return {"fd": fd, "handle": handle}

    def write_all(self, ctxs, size):
        self._batch_io(ctxs, size, opcode=1)  # CUFILE_WRITE

    def read_all(self, ctxs, size):
        self._batch_io(ctxs, size, opcode=0)  # CUFILE_READ

    def _batch_io(self, ctxs, size, opcode):
        nr = len(ctxs)
        IOParams = CUfileIOParams_t * nr
        params = IOParams()
        for i, ctx in enumerate(ctxs):
            params[i].mode = 1  # CUFILE_BATCH
            params[i].fh = ctx["handle"]
            params[i].opcode = opcode
            params[i].cookie = None
            params[i].u.batch.devPtr_base = ctx["buf_ptr"]
            params[i].u.batch.file_offset = 0
            params[i].u.batch.devPtr_offset = 0
            params[i].u.batch.size = size

        batch_handle = CUfileBatchHandle_t()
        err = libcufile.cuFileBatchIOSetUp(ctypes.byref(batch_handle), nr)
        assert err.err == CU_FILE_SUCCESS, f"cuFileBatchIOSetUp failed: {err.err}"

        err = libcufile.cuFileBatchIOSubmit(
            batch_handle, nr, params, 0
        )
        assert err.err == CU_FILE_SUCCESS, f"cuFileBatchIOSubmit failed: {err.err}"

        # Poll for completion
        events = (CUfileIOEvents_t * nr)()
        completed = ctypes.c_uint(0)
        timeout = Timespec()
        timeout.tv_sec = 10
        timeout.tv_nsec = 0
        err = libcufile.cuFileBatchIOGetStatus(
            batch_handle, nr, ctypes.byref(completed),
            events, ctypes.byref(timeout),
        )
        assert err.err == CU_FILE_SUCCESS, f"cuFileBatchIOGetStatus failed: {err.err}"
        assert completed.value == nr, f"batch incomplete: {completed.value}/{nr}"

        CUFILE_COMPLETE = 0x10
        for i in range(nr):
            assert events[i].status == CUFILE_COMPLETE, (
                f"IO {i} status=0x{events[i].status:x} ret={events[i].ret}"
            )

        libcufile.cuFileBatchIODestroy(batch_handle)

    def close(self, ctx):
        libcufile.cuFileHandleDeregister(ctx["handle"])
        os.close(ctx["fd"])


BACKENDS = {"async": AsyncBackend, "cufile": SyncBackend, "batch": BatchBackend}

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
    parser.add_argument("--verify", action="store_true",
                        help="Compare each read buffer to its corresponding write buffer")
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

    # --- Writes ---
    write_bufs = []
    for i in range(args.writes):
        buf = torch.full((aligned_size,), 0xAB, dtype=torch.uint8, device="cuda")
        err = libcufile.cuFileBufRegister(buf.data_ptr(), aligned_size, 0)
        assert err.err == CU_FILE_SUCCESS, f"cuFileBufRegister write[{i}] failed: {err.err}"
        write_bufs.append(buf)

    write_files = [os.path.join(args.dir, f"gds_bench_{i}.bin") for i in range(args.writes)]
    write_ctxs = [backend.open_file(f, os.O_CREAT | os.O_WRONLY | os.O_DIRECT) for f in write_files]
    for ctx, buf in zip(write_ctxs, write_bufs):
        ctx["buf_ptr"] = buf.data_ptr()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    backend.write_all(write_ctxs, aligned_size)

    t1 = time.perf_counter()

    for ctx in write_ctxs:
        backend.close(ctx)

    elapsed_ms = (t1 - t0) * 1000
    total_bytes = aligned_size * args.writes
    throughput_gibs = (total_bytes / (1 << 30)) / (t1 - t0) if t1 > t0 else 0
    print(f"Writes complete: {args.writes} ops, {elapsed_ms:.2f} ms, {throughput_gibs:.3f} GiB/s")

    # --- Reads ---
    read_bufs = []
    for i in range(args.reads):
        buf = torch.zeros(aligned_size, dtype=torch.uint8, device="cuda")
        err = libcufile.cuFileBufRegister(buf.data_ptr(), aligned_size, 0)
        assert err.err == CU_FILE_SUCCESS, f"cuFileBufRegister read[{i}] failed: {err.err}"
        read_bufs.append(buf)

    read_files = [os.path.join(args.dir, f"gds_bench_{i}.bin") for i in range(args.reads)]
    read_ctxs = [backend.open_file(f, os.O_RDONLY | os.O_DIRECT) for f in read_files]
    for ctx, buf in zip(read_ctxs, read_bufs):
        ctx["buf_ptr"] = buf.data_ptr()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    backend.read_all(read_ctxs, aligned_size)

    t1 = time.perf_counter()

    for ctx in read_ctxs:
        backend.close(ctx)

    elapsed_ms = (t1 - t0) * 1000
    total_bytes = aligned_size * args.reads
    throughput_gibs = (total_bytes / (1 << 30)) / (t1 - t0) if t1 > t0 else 0
    print(f"Reads complete:  {args.reads} ops, {elapsed_ms:.2f} ms, {throughput_gibs:.3f} GiB/s")

    # Verify
    if args.verify:
        num_compare = min(args.writes, args.reads)
        all_pass = True
        for i in range(num_compare):
            if not torch.equal(read_bufs[i], write_bufs[i]):
                all_pass = False
                print(f"\nVerification: FAIL (buffer {i} mismatch)")
                break
        if all_pass:
            print(f"\nVerification: PASS ({num_compare} buffers match)")
    else:
        all_pass = all(torch.all(buf == 0xAB).item() for buf in read_bufs)
        if all_pass:
            print("\nVerification: PASS")
        else:
            print("\nVerification: FAIL")

    # Cleanup
    for buf in read_bufs:
        err = libcufile.cuFileBufDeregister(buf.data_ptr())
        assert err.err == CU_FILE_SUCCESS, f"cuFileBufDeregister read failed: {err.err}"
    for buf in write_bufs:
        err = libcufile.cuFileBufDeregister(buf.data_ptr())
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
