
import multiprocessing as mp
from multiprocessing import shared_memory
from multiprocessing.util import Finalize
from array import array
import time
import statistics


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

STUDENT_ID = "230103224"

N = 384
P_PHYS = 11
P_LOG = 11

TILE_SIZES = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]

TIMED_RUNS = 3
WARMUP_RUNS = 1


# --------------------------------------------------
# WORKER GLOBAL VARIABLES
# --------------------------------------------------

SHM_A = None
SHM_B = None
SHM_C = None

A = None
B = None
C = None

_worker_finalizer = None


# --------------------------------------------------
# WORKER CLEANUP
# --------------------------------------------------

def cleanup_worker():

    global SHM_A, SHM_B, SHM_C
    global A, B, C

    # Release memory views before closing shared memory
    for view in (A, B, C):
        if view is not None:
            try:
                view.release()
            except ValueError:
                pass

    A = None
    B = None
    C = None

    # Close shared memory handles
    for shm in (SHM_A, SHM_B, SHM_C):
        if shm is not None:
            try:
                shm.close()
            except BufferError:
                pass

    SHM_A = None
    SHM_B = None
    SHM_C = None


# --------------------------------------------------
# SHARED MEMORY INITIALIZATION
# --------------------------------------------------

def initialize_worker(name_a, name_b, name_c, n):

    global SHM_A, SHM_B, SHM_C
    global A, B, C
    global _worker_finalizer

    SHM_A = shared_memory.SharedMemory(name=name_a)
    SHM_B = shared_memory.SharedMemory(name=name_b)
    SHM_C = shared_memory.SharedMemory(name=name_c)

    A = SHM_A.buf.cast("d")
    B = SHM_B.buf.cast("d")
    C = SHM_C.buf.cast("d")

    # Register cleanup when worker exits
    _worker_finalizer = Finalize(
        None,
        cleanup_worker,
        exitpriority=10
    )


# --------------------------------------------------
# VARIANT 1: NAIVE i-j-k
# --------------------------------------------------

def naive_worker(args):

    row_start, row_end = args

    for i in range(row_start, row_end):

        for j in range(N):

            dot = 0.0

            for k in range(N):

                dot += A[i * N + k] * B[k * N + j]

            C[i * N + j] = dot


# --------------------------------------------------
# VARIANT 2: REORDERED i-k-j
# --------------------------------------------------

def reordered_worker(args):

    row_start, row_end = args

    # Initialize output rows
    for i in range(row_start, row_end):

        start = i * N
        C[start:start + N] = array("d", [0.0]) * N

    for i in range(row_start, row_end):

        for k in range(N):

            a_ik = A[i * N + k]

            b_start = k * N
            c_start = i * N

            for j in range(N):

                C[c_start + j] += (
                    a_ik * B[b_start + j]
                )


# --------------------------------------------------
# VARIANT 3: CACHE-BLOCKED i-k-j
# --------------------------------------------------

def blocked_worker(args):

    row_start, row_end, BS = args

    # Initialize output rows
    for i in range(row_start, row_end):

        start = i * N
        C[start:start + N] = array("d", [0.0]) * N

    # Six-loop blocked structure:
    # ii, jj, kk, i, k, j

    for ii in range(row_start, row_end, BS):

        i_end = min(ii + BS, row_end)

        for jj in range(0, N, BS):

            j_end = min(jj + BS, N)

            for kk in range(0, N, BS):

                k_end = min(kk + BS, N)

                for i in range(ii, i_end):

                    c_start = i * N

                    for k in range(kk, k_end):

                        a_ik = A[i * N + k]
                        b_start = k * N

                        for j in range(jj, j_end):

                            C[c_start + j] += (
                                a_ik * B[b_start + j]
                            )


# --------------------------------------------------
# TASK PARTITIONING
# --------------------------------------------------

def make_row_chunks():

    chunks = []

    rows_per_worker = N // P_PHYS

    for worker in range(P_PHYS):

        start = worker * rows_per_worker

        if worker == P_PHYS - 1:
            end = N
        else:
            end = start + rows_per_worker

        chunks.append((start, end))

    return chunks


# --------------------------------------------------
# BENCHMARKING
# --------------------------------------------------

def benchmark_variant(pool, tasks, worker_function):

    # Warm-up
    for _ in range(WARMUP_RUNS):

        pool.map(worker_function, tasks)

    times = []

    # Timed measurements
    for trial in range(TIMED_RUNS):

        start_time = time.perf_counter()

        pool.map(worker_function, tasks)

        end_time = time.perf_counter()

        elapsed = end_time - start_time
        times.append(elapsed)

        print(
            f"    Trial {trial + 1}: "
            f"{elapsed * 1000:.3f} ms"
        )

    median_time = statistics.median(times)

    total_flops = 2 * (N ** 3)

    gflops = total_flops / median_time / 1e9

    return median_time, gflops


# --------------------------------------------------
# COPY SHARED MEMORY RESULT SAFELY
# --------------------------------------------------

def copy_shared_array(shm):

    view = shm.buf.cast("d")

    try:
        return array("d", view)

    finally:
        view.release()


# --------------------------------------------------
# NUMERICAL VALIDATION
# --------------------------------------------------

def compare_results(reference, candidate):

    max_difference = 0.0

    for i in range(N * N):

        difference = abs(
            reference[i] - candidate[i]
        )

        if difference > max_difference:
            max_difference = difference

    return max_difference


# --------------------------------------------------
# FNV-1a HASH
# --------------------------------------------------

def fnv1a_hash(text):

    value = 0xCBF29CE484222325

    for byte in text.encode("utf-8"):

        value ^= byte

        value = (
            value * 0x100000001B3
        ) & 0xFFFFFFFFFFFFFFFF

    return value


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=== PHASE 3B: GEMM OPTIMIZATION ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Matrix dimension: {N} x {N}")
    print(f"Physical workers: {P_PHYS}")
    print(f"Tile sizes: {TILE_SIZES}")

    size_bytes = N * N * 8

    shm_a = shared_memory.SharedMemory(
        create=True,
        size=size_bytes
    )

    shm_b = shared_memory.SharedMemory(
        create=True,
        size=size_bytes
    )

    shm_c = shared_memory.SharedMemory(
        create=True,
        size=size_bytes
    )

    try:

        # Deterministic matrix initialization
        matrix_a = array(
            "d",
            (
                ((i % 17) + 1) / 17.0
                for i in range(N * N)
            )
        )

        matrix_b = array(
            "d",
            (
                ((i % 13) + 1) / 13.0
                for i in range(N * N)
            )
        )

        shm_a.buf[:] = matrix_a.tobytes()
        shm_b.buf[:] = matrix_b.tobytes()
        shm_c.buf[:] = bytes(size_bytes)

        del matrix_a
        del matrix_b

        row_chunks = make_row_chunks()

        results = {}

        # Create persistent worker pool
        with mp.Pool(
            processes=P_PHYS,
            initializer=initialize_worker,
            initargs=(
                shm_a.name,
                shm_b.name,
                shm_c.name,
                N
            )
        ) as pool:

            # --------------------------------------
            # VARIANT 1
            # --------------------------------------

            print("\n--- Variant 1: Naive i-j-k ---")

            naive_time, naive_gflops = benchmark_variant(
                pool,
                row_chunks,
                naive_worker
            )

            naive_reference = copy_shared_array(shm_c)

            results["naive"] = (
                naive_time,
                naive_gflops
            )

            # --------------------------------------
            # VARIANT 2
            # --------------------------------------

            print("\n--- Variant 2: Reordered i-k-j ---")

            reordered_time, reordered_gflops = benchmark_variant(
                pool,
                row_chunks,
                reordered_worker
            )

            reordered_reference = copy_shared_array(shm_c)

            results["reordered"] = (
                reordered_time,
                reordered_gflops
            )

            diff_naive = compare_results(
                naive_reference,
                reordered_reference
            )

            print(
                f"\nNaive vs Reordered max difference: "
                f"{diff_naive:.12e}"
            )

            # --------------------------------------
            # VARIANT 3: TILE SIZE SWEEP
            # --------------------------------------

            print("\n--- Variant 3: Cache-Blocked i-k-j ---")

            blocked_results = {}

            for BS in TILE_SIZES:

                print(f"\nTile size BS = {BS}")

                tasks = [
                    (start, end, BS)
                    for start, end in row_chunks
                ]

                elapsed, gflops = benchmark_variant(
                    pool,
                    tasks,
                    blocked_worker
                )

                blocked_results[BS] = (
                    elapsed,
                    gflops
                )

                blocked_reference = copy_shared_array(shm_c)

                difference = compare_results(
                    reordered_reference,
                    blocked_reference
                )

                print(
                    f"    Max difference: "
                    f"{difference:.12e}"
                )

                if difference >= 1e-9:

                    raise RuntimeError(
                        f"Validation FAILED for BS={BS}: "
                        f"difference={difference}"
                    )

            results["blocked"] = blocked_results

        # ------------------------------------------
        # PERFORMANCE SUMMARY
        # ------------------------------------------

        print("\n\n=== PERFORMANCE SUMMARY ===")

        print(
            f"{'Variant':<24}"
            f"{'Median ms':>14}"
            f"{'GFLOP/s':>14}"
        )

        print("-" * 52)

        print(
            f"{'Naive i-j-k':<24}"
            f"{naive_time * 1000:>14.3f}"
            f"{naive_gflops:>14.4f}"
        )

        print(
            f"{'Reordered i-k-j':<24}"
            f"{reordered_time * 1000:>14.3f}"
            f"{reordered_gflops:>14.4f}"
        )

        print("\n--- Tile Size Sweep ---")

        print(
            f"{'BS':<10}"
            f"{'Footprint KB':>16}"
            f"{'Median ms':>14}"
            f"{'GFLOP/s':>14}"
        )

        print("-" * 54)

        for BS in TILE_SIZES:

            elapsed, gflops = blocked_results[BS]

            footprint_kb = (
                3 * BS * BS * 8 / 1024
            )

            print(
                f"{BS:<10}"
                f"{footprint_kb:>16.2f}"
                f"{elapsed * 1000:>14.3f}"
                f"{gflops:>14.4f}"
            )

        # ------------------------------------------
        # BEST TILE SIZE
        # ------------------------------------------

        best_bs = max(
            blocked_results,
            key=lambda bs: blocked_results[bs][1]
        )

        best_time, best_gflops = blocked_results[best_bs]

        print("\n=== BEST TILE SIZE ===")
        print(f"Best BS: {best_bs}")
        print(f"Best median time: {best_time * 1000:.3f} ms")
        print(f"Best throughput: {best_gflops:.4f} GFLOP/s")

        # ------------------------------------------
        # VALIDATION SUMMARY
        # ------------------------------------------

        print("\n=== VALIDATION ===")

        if diff_naive < 1e-9:
            print("Naive vs Reordered: PASS")
        else:
            print("Naive vs Reordered: FAIL")

        print("All blocked variants: PASS")

        # ------------------------------------------
        # PHASE 3 HASH
        # ------------------------------------------

        payload = (
            f"PHASE3|{STUDENT_ID}|{P_LOG}|"
            f"{naive_gflops:.4f}|{best_gflops:.4f}"
        )

        signature = fnv1a_hash(payload)

        print(
            f"\nVERIFY_HASH [PHASE3]: "
            f"{signature:016X} (P_log={P_LOG})"
        )

    finally:

        # Parent process releases its own shared memory
        shm_a.close()
        shm_b.close()
        shm_c.close()

        shm_a.unlink()
        shm_b.unlink()
        shm_c.unlink()


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    mp.freeze_support()
    main()
