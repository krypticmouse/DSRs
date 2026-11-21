"""Replay paired runtime measurements with two binaries built from harness.rs."""

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path


WORKLOADS = {"skewed": [64], "uniform": [64], "preflight": [4096, 32768]}
ORDER = ("baseline", "candidate", "candidate", "baseline")
MAX_PAIRED_RATIO = {"skewed": 0.80, "uniform": 1.10, "preflight": 0.80, "score-only": 0.85}


def measure(executables, output, workloads):
    summaries = {}
    columns = ["workload", "size", "process", "build", "warmup", "elapsed_us", "exit"]
    with (output / "samples.csv").open("x", newline="", encoding="utf-8") as file:
        raw = csv.writer(file, lineterminator="\n")
        raw.writerow(columns)
        for workload, sizes in workloads.items():
            timings = {size: {label: [] for label in executables} for size in sizes}
            for process, label in enumerate(("baseline", "candidate") + ORDER * 4, start=1):
                warmup = process <= 2
                result = subprocess.run(
                    [executables[label], workload],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode or result.stderr:
                    log = output / f"{workload}-{process}.stderr.txt"
                    log.write_text(result.stderr, encoding="utf-8")
                seen = []
                for line in result.stdout.splitlines():
                    observed_workload, size, micros = line.split(",")
                    size, micros = int(size), int(micros)
                    if observed_workload != workload or micros <= 0:
                        raise RuntimeError(f"invalid output from {workload} process {process}")
                    seen.append(size)
                    raw.writerow([workload, size, process, label, warmup, micros, result.returncode])
                    if not warmup and result.returncode == 0:
                        timings[size][label].append(micros)
                file.flush()
                if result.returncode or seen != sizes:
                    raise RuntimeError(f"{workload} process {process} did not complete")

            for size, series in timings.items():
                baseline = series["baseline"]
                candidate = series["candidate"]
                ratios = [after / before for before, after in zip(baseline, candidate)]
                paired_ratio = statistics.median(ratios)
                summaries[f"{workload}/{size}"] = {
                    "baseline_median_us": statistics.median(baseline),
                    "baseline_range_us": [min(baseline), max(baseline)],
                    "candidate_median_us": statistics.median(candidate),
                    "candidate_range_us": [min(candidate), max(candidate)],
                    "median_paired_ratio": paired_ratio,
                    "faster_pairs": sum(ratio < 1 for ratio in ratios),
                    "maximum_paired_ratio": MAX_PAIRED_RATIO[workload],
                    "passed": paired_ratio <= MAX_PAIRED_RATIO[workload],
                }
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-only", action="store_true", help="measure evaluator output conversion")
    options = parser.parse_args()
    executables = {
        "baseline": options.baseline.resolve(),
        "candidate": options.candidate.resolve(),
    }
    hashes = {label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in executables.items()}
    if hashes["baseline"] == hashes["candidate"]:
        parser.error("baseline and candidate must be distinct builds")
    workloads = {"score-only": [64, 4096, 65536]} if options.score_only else WORKLOADS
    options.output.mkdir(parents=True)
    summary = {
        "samples_per_binary": 8,
        "sha256": hashes,
        "cases": measure(executables, options.output, workloads),
    }
    payload = json.dumps(summary, indent=2)
    (options.output / "summary.json").write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
