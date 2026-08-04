#!/usr/bin/env python
"""rebuttal — occupy specified GPUs (fill memory + keep utilization ~100%).

Usage:
    python rebuttal.py 4 6              # occupy GPU 4 and GPU 6
    python rebuttal.py 4,6              # same, comma-separated also works
    python rebuttal.py 4 6 --mem 0.8    # only grab 80% of currently-free memory
    python rebuttal.py 4 6 --size 4096  # smaller matmul (lower power draw)

The process renames itself to 'rebuttal' (via a symlink to the python binary),
so nvidia-smi / top show 'rebuttal' instead of 'python'.

Stop with Ctrl+C, or from another shell:  pkill -x rebuttal
(exact-name match only — will never touch other python/ray processes)
"""
import argparse
import os
import signal
import sys
import threading
import time

PROC_NAME = "rebuttal_exp"


def ensure_process_name():
    """Re-exec through a symlink named 'rebuttal' so argv[0]/nvidia-smi show it."""
    if os.path.basename(sys.executable) == PROC_NAME:
        return
    link = os.path.join(os.path.dirname(os.path.abspath(__file__)), PROC_NAME)
    real_python = os.path.realpath(sys.executable)
    try:
        if not (os.path.islink(link) and os.path.realpath(link) == real_python):
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(real_python, link)
        os.execv(link, [link] + sys.argv)
    except OSError as e:
        print(f"[rebuttal] warning: could not rename process ({e}); "
              f"continuing as {os.path.basename(sys.executable)}", file=sys.stderr)


def set_thread_comm():
    """Set the kernel-level task name (what `top` and `pkill -x` see)."""
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(15, PROC_NAME.encode(), 0, 0, 0)  # PR_SET_NAME
    except Exception:
        pass


def parse_gpus(tokens):
    gpus = []
    for tok in tokens:
        for part in tok.split(","):
            part = part.strip()
            if part:
                gpus.append(int(part))
    return sorted(set(gpus))


def occupy(gpu_id, mem_frac, size, stop_event):
    try:
        _occupy(gpu_id, mem_frac, size, stop_event)
    except Exception as e:
        print(f"[rebuttal] GPU {gpu_id} worker died: {e!r}", file=sys.stderr, flush=True)


def _occupy(gpu_id, mem_frac, size, stop_event):
    import torch
    set_thread_comm()
    dev = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(dev)

    # compute tensors first, then fill the rest of free memory with ballast
    a = torch.randn(size, size, device=dev, dtype=torch.float16)
    b = torch.randn(size, size, device=dev, dtype=torch.float16)
    c = torch.empty(size, size, device=dev, dtype=torch.float16)

    ballast = []
    block = 512 * 1024 * 1024  # 512 MiB
    free0, _ = torch.cuda.mem_get_info(dev)
    headroom = int((1.0 - mem_frac) * free0)  # leave this much untouched
    while True:
        free, _ = torch.cuda.mem_get_info(dev)
        if free - block < headroom:
            break
        try:
            ballast.append(torch.empty(block // 2, dtype=torch.float16, device=dev))
        except torch.cuda.OutOfMemoryError:
            break
    used = torch.cuda.memory_allocated(dev) / 1024**3
    print(f"[rebuttal] GPU {gpu_id}: holding {used:.1f} GiB, running matmul loop",
          flush=True)

    i = 0
    while not stop_event.is_set():
        torch.matmul(a, b, out=c)
        i += 1
        if i % 200 == 0:
            torch.cuda.synchronize(dev)
    torch.cuda.synchronize(dev)


def main():
    ensure_process_name()
    parser = argparse.ArgumentParser(prog=PROC_NAME, description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gpus", nargs="+", help="GPU ids, e.g. '4 6' or '4,6'")
    parser.add_argument("--mem", type=float, default=0.95,
                        help="fraction of currently-free memory to hold (default 0.95)")
    parser.add_argument("--size", type=int, default=8192,
                        help="matmul dimension; larger = higher power draw (default 8192)")
    args = parser.parse_args()
    gpus = parse_gpus(args.gpus)

    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit(f"[rebuttal] error: this python ({sys.executable}) has no torch — "
                 f"run with the 'duet' conda env python")

    set_thread_comm()
    stop_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())

    print(f"[rebuttal] pid {os.getpid()} occupying GPUs {gpus} "
          f"(mem={args.mem:.0%}, matmul {args.size}x{args.size})", flush=True)
    threads = [threading.Thread(target=occupy, args=(g, args.mem, args.size, stop_event),
                                daemon=True, name=f"gpu{g}") for g in gpus]
    for t in threads:
        t.start()
    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
    for t in threads:
        t.join(timeout=10)
    print("[rebuttal] released all GPUs, bye")


if __name__ == "__main__":
    main()
