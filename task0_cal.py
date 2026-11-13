import multiprocessing as mp
import time

from harness import emit_verify_tag


STUDENT_ID = 230103224
NUM_READS = 2_000_000


def main():
    logical_cores = mp.cpu_count()

    start = time.perf_counter_ns()

    for _ in range(NUM_READS):
        time.perf_counter_ns()

    end = time.perf_counter_ns()

    total_time_ns = end - start
    average_latency_ns = total_time_ns / NUM_READS
    total_time_ms = total_time_ns / 1_000_000

    print("=== PHASE 0: CALIBRATION ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical CPU count: {logical_cores}")
    print(f"Clock reads: {NUM_READS}")
    print(f"Total time: {total_time_ms:.4f} ms")
    print(f"Average clock read latency: {average_latency_ns:.4f} ns")

    print(
        emit_verify_tag(
            "PHASE0",
            STUDENT_ID,
            average_latency_ns,
            total_time_ms
        )
    )


if __name__ == "__main__":
    main()