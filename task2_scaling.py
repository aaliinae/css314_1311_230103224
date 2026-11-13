
import multiprocessing as mp
import math
import time
import statistics
from harness import emit_verify_tag


# ==========================================
# PHASE 2: STRONG AND WEAK SCALING
# ==========================================

STUDENT_ID = 230103224

P_PHYS = 11
P_LOG = 11

N_STRONG = 1_600_000
N0_WEAK = 250_000

SERIAL_FRACTION_ELEMENTS = 0.05
STAGE_A_OPERATIONS = 12
STAGE_B_OPERATIONS = 35

WARMUP_RUNS = 0
TIMED_RUNS = 3

PROCESS_COUNTS = list(range(1, P_LOG + 1))


# ==========================================
# STAGE A: SERIAL COMPUTATION
# ==========================================

def stage_a(n):
    count = int(n * SERIAL_FRACTION_ELEMENTS)
    total = 0.0

    for i in range(count):
        x = (i + 1) * 0.00001

        for j in range(STAGE_A_OPERATIONS):
            x = math.sin(x) + math.sqrt(abs(x) + 1e-12)

        total += x

    return total


# ==========================================
# STAGE B: PARALLEL COMPUTATION
# ==========================================

def stage_b_chunk(args):
    start, end = args
    total = 0.0

    for i in range(start, end):
        x = (i + 1) * 0.00001

        for j in range(STAGE_B_OPERATIONS):
            x = math.sin(x * 1.000001 + 0.000001)

        total += x

    return total


def make_chunks(n, p):
    chunks = []

    for worker in range(p):
        start = worker * n // p
        end = (worker + 1) * n // p

        chunks.append((start, end))

    return chunks


# ==========================================
# ONE EXPERIMENTAL TRIAL
# ==========================================

def run_trial(n, pool, p):
    total_start = time.perf_counter_ns()

    # Serial Stage A
    stage_a_start = time.perf_counter_ns()

    serial_result = stage_a(n)

    stage_a_end = time.perf_counter_ns()

    # Parallel Stage B
    chunks = make_chunks(n, p)

    parallel_result = pool.map(stage_b_chunk, chunks)

    total_end = time.perf_counter_ns()

    stage_a_ms = (
        stage_a_end - stage_a_start
    ) / 1e6

    total_ms = (
        total_end - total_start
    ) / 1e6

    checksum = serial_result + sum(parallel_result)

    return total_ms, stage_a_ms, checksum


# ==========================================
# BENCHMARK
# ==========================================

def benchmark(n, p, experiment_name):
    print(
        f"\nRunning {experiment_name}: "
        f"N={n:,}, processes={p}"
    )

    total_times = []
    serial_times = []
    checksums = []

    # Pool creation is outside the timed region.
    with mp.Pool(processes=p) as pool:

        for _ in range(WARMUP_RUNS):
            run_trial(n, pool, p)

        for trial in range(TIMED_RUNS):
            total_ms, stage_a_ms, checksum = run_trial(
                n, pool, p
            )

            total_times.append(total_ms)
            serial_times.append(stage_a_ms)
            checksums.append(checksum)

            print(
                f"  Trial {trial + 1}: "
                f"Total={total_ms:.3f} ms, "
                f"Stage A={stage_a_ms:.3f} ms"
            )

    return {
        "median_ms": statistics.median(total_times),
        "stage_a_ms": statistics.median(serial_times),
        "checksum": statistics.median(checksums),
        "samples": total_times
    }


# ==========================================
# STRONG SCALING
# ==========================================

def strong_scaling():
    print("\n" + "=" * 65)
    print("EXPERIMENT 2A: STRONG SCALING")
    print("=" * 65)

    results = {}

    for p in PROCESS_COUNTS:
        results[p] = benchmark(
            N_STRONG,
            p,
            "STRONG"
        )

    t1 = results[1]["median_ms"]

    print("\nSTRONG SCALING RESULTS")
    print(
        f"{'P':>4} {'Time(ms)':>14} "
        f"{'Speedup':>12} {'Efficiency':>12}"
    )

    for p in PROCESS_COUNTS:
        tp = results[p]["median_ms"]

        speedup = t1 / tp
        efficiency = speedup / p

        results[p]["speedup"] = speedup
        results[p]["efficiency"] = efficiency

        print(
            f"{p:4d} {tp:14.3f} "
            f"{speedup:12.4f} {efficiency:12.4f}"
        )

    return results


# ==========================================
# WEAK SCALING
# ==========================================

def weak_scaling():
    print("\n" + "=" * 65)
    print("EXPERIMENT 2B: WEAK SCALING")
    print("=" * 65)

    results = {}

    for p in PROCESS_COUNTS:
        n = p * N0_WEAK

        results[p] = benchmark(
            n,
            p,
            "WEAK"
        )

    t1 = results[1]["median_ms"]

    print("\nWEAK SCALING RESULTS")
    print(
        f"{'P':>4} {'N':>12} {'Time(ms)':>14} "
        f"{'Weak Eff.':>12} {'Gustafson':>12}"
    )

    for p in PROCESS_COUNTS:
        tp = results[p]["median_ms"]
        stage_a_time = results[p]["stage_a_ms"]

        weak_efficiency = t1 / tp

        epsilon = stage_a_time / tp

        gustafson_speedup = p - epsilon * (p - 1)

        results[p]["weak_efficiency"] = weak_efficiency
        results[p]["gustafson_speedup"] = gustafson_speedup
        results[p]["epsilon"] = epsilon

        print(
            f"{p:4d} {p * N0_WEAK:12,} "
            f"{tp:14.3f} "
            f"{weak_efficiency:12.4f} "
            f"{gustafson_speedup:12.4f}"
        )

    return results


# ==========================================
# AMDAHL'S LAW
# ==========================================

def amdahl_analysis(strong):
    print("\n" + "=" * 65)
    print("AMDAHL'S LAW ANALYSIS")
    print("=" * 65)

    t1 = strong[1]["median_ms"]

    serial_time = strong[1]["stage_a_ms"]

    s = serial_time / t1

    theoretical_max = 1 / s

    p = P_LOG

    theoretical_speedup = 1 / (
        s + (1 - s) / p
    )

    observed_speedup = strong[p]["speedup"]

    print(f"Serial time: {serial_time:.4f} ms")
    print(f"Total time at p=1: {t1:.4f} ms")
    print(f"Serial fraction (s): {s:.6f}")
    print(f"Amdahl maximum speedup: {theoretical_max:.4f}")
    print(
        f"Theoretical speedup at p={p}: "
        f"{theoretical_speedup:.4f}"
    )
    print(
        f"Observed speedup at p={p}: "
        f"{observed_speedup:.4f}"
    )

    return s, theoretical_max, theoretical_speedup


# ==========================================
# VERIFICATION HASH
# ==========================================

def verification(strong, weak):
    strong_speedup = strong[P_LOG]["speedup"]

    gustafson_speedup = weak[P_LOG]["gustafson_speedup"]

    print("\n" + "=" * 65)
    print("PHASE 2 VERIFICATION")
    print("=" * 65)

    print(
        emit_verify_tag(
            "PHASE2",
            STUDENT_ID,
            strong_speedup,
            gustafson_speedup
        )
    )


# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 65)
    print("PHASE 2: STRONG AND WEAK SCALING")
    print("=" * 65)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Physical cores: {P_PHYS}")
    print(f"Logical cores: {P_LOG}")
    print(f"Strong N: {N_STRONG:,}")
    print(f"Weak N0: {N0_WEAK:,}")
    print(f"Timed runs: {TIMED_RUNS}")

    strong = strong_scaling()

    weak = weak_scaling()

    amdahl_analysis(strong)

    verification(strong, weak)

    print("\nPHASE 2 COMPLETED")


if __name__ == "__main__":
    mp.freeze_support()
    main()
