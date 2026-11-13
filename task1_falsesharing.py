
import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import time

from harness import run_benchmark, emit_verify_tag


STUDENT_ID = 230103224
M = 24_000_000
SHM_SIZE = 4096
STRIDES = [1, 2, 4, 8, 16, 32]
P = 11

MASK = 0xFFFFFFFFFFFFFFFF
MULT = 6364136223846793005


def fs_worker(shm_name, slot_idx, iters):
    shm = shared_memory.SharedMemory(name=shm_name)

    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

    acc = arr[slot_idx]

    for i in range(iters):
        acc = (acc * MULT + (i | 1)) & MASK
        arr[slot_idx] = acc

    del arr
    shm.close()


def run_false_sharing(stride, p):
    shm = shared_memory.SharedMemory(
        create=True,
        size=SHM_SIZE
    )

    processes = []

    try:
        arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

        for i in range(512):
            arr[i] = 0

        del arr

        iterations = M // p

        start = time.perf_counter_ns()

        for t in range(p):
            process = mp.Process(
                target=fs_worker,
                args=(shm.name, t * stride, iterations)
            )

            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        elapsed = time.perf_counter_ns() - start

        for process in processes:
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Worker failed: exit code {process.exitcode}"
                )

        return elapsed / 1e6

    finally:
        shm.close()
        shm.unlink()


def main():
    print("=== PHASE 1: FALSE SHARING ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Physical processes: {P}")
    print(f"Total iterations: {M}")

    results = {}

    print("\n--- STRIDE SWEEP ---")

    for stride in STRIDES:
        stats = run_benchmark(
            lambda s=stride: run_false_sharing(s, P),
            warmup_runs=2,
            timed_runs=7
        )

        throughput = M / (stats.median_ms * 1000)

        results[stride] = stats

        print(f"\nStride: {stride}")
        print(f"Separation: {stride * 8} bytes")
        print(f"Median: {stats.median_ms:.4f} ms")
        print(f"IQR: {stats.iqr_ms:.4f} ms")
        print(f"Throughput: {throughput:.4f} Mops/s")

    print("\n--- PHASE 1 VERIFICATION ---")

    print(
        emit_verify_tag(
            "PHASE1",
            STUDENT_ID,
            results[1].median_ms,
            results[16].median_ms
        )
    )

    print("\n--- PROCESS SCALING ---")

    process_counts = [1, 2, 4, 8, 11]

    for stride in [1, 16]:
        print(f"\n=== STRIDE {stride} ===")

        scaling_results = {}

        for p in process_counts:
            stats = run_benchmark(
                lambda s=stride, workers=p:
                    run_false_sharing(s, workers),
                warmup_runs=2,
                timed_runs=7
            )

            scaling_results[p] = stats.median_ms

        t1 = scaling_results[1]

        print(
            f"{'P':>5} {'Time(ms)':>12} "
            f"{'Speedup':>12} {'Efficiency':>12} "
            f"{'Karp-Flatt':>12}"
        )

        for p in process_counts:
            tp = scaling_results[p]
            speedup = t1 / tp
            efficiency = speedup / p

            if p == 1:
                kf_text = "N/A"
            else:
                karp_flatt = (
                    (1 / speedup - 1 / p) /
                    (1 - 1 / p)
                )
                kf_text = f"{karp_flatt:.4f}"

            print(
                f"{p:5d} {tp:12.4f} "
                f"{speedup:12.4f} "
                f"{efficiency:12.4f} "
                f"{kf_text:>12}"
            )


if __name__ == "__main__":
    mp.freeze_support()
    main()

    
