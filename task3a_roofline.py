
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import statistics
import hashlib
import struct


STUDENT_ID = "230103224"

P_PHYS = 11
P_LOG = 11

BUFFER_SIZE = 96 * 1024 * 1024  # 96 MB
FLOPS_PER_WORKER = 80_000_000

TIMED_RUNS = 3


# --------------------------------------------------
# MEMORY BANDWIDTH WORKER
# --------------------------------------------------

def memory_copy_worker(args):
    shm_a_name, shm_b_name, start, end = args

    shm_a = shared_memory.SharedMemory(name=shm_a_name)
    shm_b = shared_memory.SharedMemory(name=shm_b_name)

    try:
        mv_a = shm_a.buf
        mv_b = shm_b.buf

        mv_a[start:end] = mv_b[start:end]

    finally:
        shm_a.close()
        shm_b.close()


def measure_bandwidth():

    shm_a = shared_memory.SharedMemory(
        create=True,
        size=BUFFER_SIZE
    )

    shm_b = shared_memory.SharedMemory(
        create=True,
        size=BUFFER_SIZE
    )

    try:
        # Initialize source buffer
        shm_b.buf[:] = b"\x5A" * BUFFER_SIZE

        chunk_size = BUFFER_SIZE // P_PHYS

        chunks = []

        for i in range(P_PHYS):
            start = i * chunk_size

            if i == P_PHYS - 1:
                end = BUFFER_SIZE
            else:
                end = start + chunk_size

            chunks.append((
                shm_a.name,
                shm_b.name,
                start,
                end
            ))

        times = []

        # IMPORTANT:
        # Create Pool ONCE, outside the measurement loop.
        with mp.Pool(processes=P_PHYS) as pool:

            # Warm-up
            pool.map(memory_copy_worker, chunks)

            # Timed measurements
            for trial in range(TIMED_RUNS):

                start_time = time.perf_counter()

                pool.map(memory_copy_worker, chunks)

                end_time = time.perf_counter()

                elapsed = end_time - start_time
                times.append(elapsed)

                print(
                    f"Bandwidth trial {trial + 1}: "
                    f"{elapsed:.6f} sec"
                )

        median_time = statistics.median(times)

        # Each byte is read once and written once.
        total_bytes = 2 * BUFFER_SIZE

        bandwidth = (
            total_bytes / median_time / 1e9
        )

        return bandwidth, median_time

    finally:
        shm_a.close()
        shm_b.close()

        shm_a.unlink()
        shm_b.unlink()


# --------------------------------------------------
# COMPUTE WORKER
# --------------------------------------------------

def compute_worker(iterations):

    a = 1.0000001
    b = 0.0000001

    x0 = 1.0
    x1 = 1.1
    x2 = 1.2
    x3 = 1.3
    x4 = 1.4
    x5 = 1.5
    x6 = 1.6
    x7 = 1.7
    x8 = 1.8
    x9 = 1.9
    x10 = 2.0
    x11 = 2.1
    x12 = 2.2
    x13 = 2.3
    x14 = 2.4
    x15 = 2.5

    for _ in range(iterations):

        x0 = x0 * a + b
        x1 = x1 * a + b
        x2 = x2 * a + b
        x3 = x3 * a + b
        x4 = x4 * a + b
        x5 = x5 * a + b
        x6 = x6 * a + b
        x7 = x7 * a + b
        x8 = x8 * a + b
        x9 = x9 * a + b
        x10 = x10 * a + b
        x11 = x11 * a + b
        x12 = x12 * a + b
        x13 = x13 * a + b
        x14 = x14 * a + b
        x15 = x15 * a + b

    return (
        x0 + x1 + x2 + x3 +
        x4 + x5 + x6 + x7 +
        x8 + x9 + x10 + x11 +
        x12 + x13 + x14 + x15
    )


def measure_compute():

    iterations = FLOPS_PER_WORKER // 32

    times = []

    # IMPORTANT:
    # Create Pool ONCE, outside the measurement loop.
    with mp.Pool(processes=P_PHYS) as pool:

        # Warm-up
        pool.map(
            compute_worker,
            [iterations] * P_PHYS
        )

        # Timed measurements
        for trial in range(TIMED_RUNS):

            start_time = time.perf_counter()

            results = pool.map(
                compute_worker,
                [iterations] * P_PHYS
            )

            end_time = time.perf_counter()

            elapsed = end_time - start_time
            times.append(elapsed)

            print(
                f"Compute trial {trial + 1}: "
                f"{elapsed:.6f} sec"
            )

    median_time = statistics.median(times)

    total_flops = FLOPS_PER_WORKER * P_PHYS

    gflops = (
        total_flops / median_time / 1e9
    )

    return gflops, median_time


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=== PHASE 3A: ROOFLINE CALIBRATION ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Physical cores: {P_PHYS}")
    print(f"Logical cores: {P_LOG}")

    print("\n--- Memory Bandwidth ---")

    bandwidth, bandwidth_time = measure_bandwidth()

    print(
        f"Sustained bandwidth: "
        f"{bandwidth:.4f} GB/s"
    )

    print(
        f"Median bandwidth time: "
        f"{bandwidth_time:.6f} sec"
    )

    print("\n--- Compute Ceiling ---")

    compute, compute_time = measure_compute()

    print(
        f"Compute ceiling: "
        f"{compute:.4f} GFLOP/s"
    )

    print(
        f"Median compute time: "
        f"{compute_time:.6f} sec"
    )

    ridge = compute / bandwidth

    print(
        f"\nRidge point: "
        f"{ridge:.4f} FLOPs/Byte"
    )

    # Reproducibility hash based on measured results
    hash_data = struct.pack(
        "ddd",
        bandwidth,
        compute,
        ridge
    )

    verify_hash = hashlib.sha256(
        hash_data
    ).hexdigest()[:16].upper()

    print(
        f"\nVERIFY_HASH [PHASE3A]: "
        f"{verify_hash} (P_log={P_LOG})"
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
