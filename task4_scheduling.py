
import multiprocessing as mp
import math
import time
import statistics


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

STUDENT_ID = "230103224"

N = 12000
P_PHYS = 11
P_LOG = 11

CHUNK_SIZES = [1, 4, 16, 64, 256, 1024]


# --------------------------------------------------
# WORKLOAD
# --------------------------------------------------

def workload(index):

    # Number of iterations for this item
    iterations = 1 + math.ceil((index / 60) ** 2)

    x = 1.0 + (index % 10) * 0.01

    for _ in range(iterations):

        x = (
            math.sin(x) * math.cos(x)
            + math.sqrt(abs(x) + 1.0)
        )

    return x


# --------------------------------------------------
# SHARED WORKER METRICS
# --------------------------------------------------

def record_worker_result(worker_id, start_time, times, checksums, checksum):

    end_time = time.perf_counter()

    times[worker_id] = end_time - start_time
    checksums[worker_id] = checksum


# --------------------------------------------------
# STRATEGY 1: CONTIGUOUS BLOCK STATIC
# --------------------------------------------------

def static_block_worker(worker_id, start, end, times, checksums):

    start_time = time.perf_counter()

    checksum = 0.0

    for i in range(start, end):

        checksum += workload(i)

    record_worker_result(
        worker_id,
        start_time,
        times,
        checksums,
        checksum
    )


def run_static_block():

    times = mp.Array("d", P_PHYS, lock=False)
    checksums = mp.Array("d", P_PHYS, lock=False)

    processes = []

    rows_per_worker = N // P_PHYS

    wall_start = time.perf_counter()

    for worker_id in range(P_PHYS):

        start = worker_id * rows_per_worker

        if worker_id == P_PHYS - 1:
            end = N
        else:
            end = start + rows_per_worker

        process = mp.Process(
            target=static_block_worker,
            args=(
                worker_id,
                start,
                end,
                times,
                checksums
            )
        )

        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    wall_time = time.perf_counter() - wall_start

    worker_times = list(times)

    omega = (
        max(worker_times)
        / statistics.mean(worker_times)
    )

    return wall_time, omega, worker_times


# --------------------------------------------------
# STRATEGY 2: CYCLIC ROUND-ROBIN
# --------------------------------------------------

def cyclic_worker(worker_id, times, checksums):

    start_time = time.perf_counter()

    checksum = 0.0

    for i in range(worker_id, N, P_PHYS):

        checksum += workload(i)

    record_worker_result(
        worker_id,
        start_time,
        times,
        checksums,
        checksum
    )


def run_cyclic():

    times = mp.Array("d", P_PHYS, lock=False)
    checksums = mp.Array("d", P_PHYS, lock=False)

    processes = []

    wall_start = time.perf_counter()

    for worker_id in range(P_PHYS):

        process = mp.Process(
            target=cyclic_worker,
            args=(
                worker_id,
                times,
                checksums
            )
        )

        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    wall_time = time.perf_counter() - wall_start

    worker_times = list(times)

    omega = (
        max(worker_times)
        / statistics.mean(worker_times)
    )

    return wall_time, omega, worker_times


# --------------------------------------------------
# STRATEGY 3: DYNAMIC CHUNK QUEUE
# --------------------------------------------------

def dynamic_worker(
    worker_id,
    chunk_size,
    counter,
    times,
    checksums
):

    start_time = time.perf_counter()

    checksum = 0.0

    while True:

        # Claim the next chunk safely
        with counter.get_lock():

            start = counter.value

            counter.value += chunk_size

        if start >= N:
            break

        end = min(start + chunk_size, N)

        for i in range(start, end):

            checksum += workload(i)

    record_worker_result(
        worker_id,
        start_time,
        times,
        checksums,
        checksum
    )


def run_dynamic(chunk_size):

    times = mp.Array("d", P_PHYS, lock=False)
    checksums = mp.Array("d", P_PHYS, lock=False)

    counter = mp.Value("i", 0)

    processes = []

    wall_start = time.perf_counter()

    for worker_id in range(P_PHYS):

        process = mp.Process(
            target=dynamic_worker,
            args=(
                worker_id,
                chunk_size,
                counter,
                times,
                checksums
            )
        )

        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    wall_time = time.perf_counter() - wall_start

    worker_times = list(times)

    omega = (
        max(worker_times)
        / statistics.mean(worker_times)
    )

    return wall_time, omega, worker_times


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
# PRINT RESULTS
# --------------------------------------------------

def print_result(name, wall_time, omega):

    print(
        f"{name:<25}"
        f"{wall_time * 1000:>14.3f}"
        f"{omega:>14.4f}"
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=== PHASE 4: SCHEDULING & LOAD IMBALANCE ===")

    print(f"Student ID: {STUDENT_ID}")
    print(f"Total workload items: {N}")
    print(f"Physical workers: {P_PHYS}")
    print(f"Dynamic chunk sizes: {CHUNK_SIZES}")

    # ------------------------------------------
    # STRATEGY 1
    # ------------------------------------------

    print("\n--- Strategy 1: Contiguous Block Static ---")

    static_time, static_omega, static_workers = (
        run_static_block()
    )

    print_result(
        "Static Block",
        static_time,
        static_omega
    )

    # ------------------------------------------
    # STRATEGY 2
    # ------------------------------------------

    print("\n--- Strategy 2: Cyclic Round-Robin ---")

    cyclic_time, cyclic_omega, cyclic_workers = (
        run_cyclic()
    )

    print_result(
        "Cyclic",
        cyclic_time,
        cyclic_omega
    )

    # ------------------------------------------
    # STRATEGY 3
    # ------------------------------------------

    print("\n--- Strategy 3: Dynamic Chunk Queue ---")

    dynamic_results = {}

    for chunk_size in CHUNK_SIZES:

        print(f"\nChunk size C = {chunk_size}")

        wall_time, omega, worker_times = run_dynamic(
            chunk_size
        )

        dynamic_results[chunk_size] = (
            wall_time,
            omega,
            worker_times
        )

        print_result(
            f"Dynamic C={chunk_size}",
            wall_time,
            omega
        )

    # ------------------------------------------
    # SUMMARY
    # ------------------------------------------

    print("\n\n=== PERFORMANCE SUMMARY ===")

    print(
        f"{'Strategy':<25}"
        f"{'Wall time ms':>14}"
        f"{'Omega':>14}"
    )

    print("-" * 53)

    print_result(
        "Static Block",
        static_time,
        static_omega
    )

    print_result(
        "Cyclic",
        cyclic_time,
        cyclic_omega
    )

    for chunk_size in CHUNK_SIZES:

        wall_time, omega, _ = dynamic_results[chunk_size]

        print_result(
            f"Dynamic C={chunk_size}",
            wall_time,
            omega
        )

    # ------------------------------------------
    # BEST CHUNK SIZE
    # ------------------------------------------

    best_chunk = min(
        dynamic_results,
        key=lambda c: dynamic_results[c][0]
    )

    best_time, best_omega, best_workers = (
        dynamic_results[best_chunk]
    )

    print("\n=== BEST DYNAMIC CONFIGURATION ===")

    print(f"Best chunk size: {best_chunk}")
    print(f"Best wall time: {best_time * 1000:.3f} ms")
    print(f"Best omega: {best_omega:.4f}")

    # ------------------------------------------
    # WORKER TIME DETAILS
    # ------------------------------------------

    print("\n=== WORKER ACTIVE TIMES ===")

    print("\nStatic Block:")

    for i, value in enumerate(static_workers):

        print(
            f"Worker {i}: {value * 1000:.3f} ms"
        )

    print("\nCyclic:")

    for i, value in enumerate(cyclic_workers):

        print(
            f"Worker {i}: {value * 1000:.3f} ms"
        )

    print(f"\nDynamic C={best_chunk}:")

    for i, value in enumerate(best_workers):

        print(
            f"Worker {i}: {value * 1000:.3f} ms"
        )

    # ------------------------------------------
    # PHASE 4 HASH
    # ------------------------------------------

    payload = (
        f"PHASE4|{STUDENT_ID}|{P_LOG}|"
        f"{static_time:.4f}|{best_time:.4f}"
    )

    signature = fnv1a_hash(payload)

    print(
        f"\nVERIFY_HASH [PHASE4]: "
        f"{signature:016X} (P_log={P_LOG})"
    )


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    mp.freeze_support()
    main()
