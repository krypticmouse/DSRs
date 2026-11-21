import csv
import ctypes as c
from ctypes import wintypes
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


ROOT = Path(__file__).resolve().parent
ORDER = ("baseline", "trial", "trial", "baseline") * 4
MODES = {
    "cache": {"cache": 8192},
    "preflight": {"preflight": 32768},
    "jsonl": {"jsonl": 8192},
}


class MemoryCounters(c.Structure):
    _fields_ = [
        ("size", wintypes.DWORD),
        ("page_fault_count", wintypes.DWORD),
        ("peak_working_set_bytes", c.c_size_t),
        ("working_set_bytes", c.c_size_t),
        ("peak_paged_pool_bytes", c.c_size_t),
        ("paged_pool_bytes", c.c_size_t),
        ("peak_nonpaged_pool_bytes", c.c_size_t),
        ("nonpaged_pool_bytes", c.c_size_t),
        ("pagefile_bytes", c.c_size_t),
        ("peak_pagefile_bytes", c.c_size_t),
    ]


memory_info = c.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
memory_info.argtypes = [wintypes.HANDLE, c.POINTER(MemoryCounters), wintypes.DWORD]
memory_info.restype = wintypes.BOOL


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_fixture(path):
    text = b"x" * 16384
    with path.open("wb") as file:
        for index in range(8192):
            file.write(b'{"id":' + str(index).encode() + b',"text":"' + text + b'"}\n')


def run_jsonl(executable, fixture):
    args = [executable, "jsonl", fixture]
    with ThreadPoolExecutor(max_workers=1) as reader, subprocess.Popen(
        args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as process:
        first_line = reader.submit(process.stdout.readline)
        try:
            first = first_line.result(timeout=60)
            counters = MemoryCounters()
            counters.size = c.sizeof(counters)
            peak = None
            if first:
                handle = wintypes.HANDLE(int(process._handle))
                if not memory_info(handle, c.byref(counters), counters.size):
                    raise c.WinError(c.get_last_error())
                peak = counters.peak_working_set_bytes
                process.stdin.write("done\n")
                process.stdin.flush()
            stdout, stderr = process.communicate(timeout=60)
        except BaseException:
            process.kill()
            process.wait()
            raise
    return subprocess.CompletedProcess(args, process.returncode, first + stdout, stderr), peak


def run_case(mode, build, ordinal, warmup, fixture):
    executable = ROOT / f"{build}.exe"
    stem = ROOT / "runs" / f"{mode}-{ordinal:02}-{build}"
    if mode == "jsonl":
        process, peak = run_jsonl(executable, fixture)
    else:
        process = subprocess.run([executable, mode], capture_output=True, text=True, timeout=60)
        peak = None
    stem.with_suffix(".stdout").write_text(process.stdout, encoding="utf-8")
    stem.with_suffix(".stderr").write_text(process.stderr, encoding="utf-8")
    stem.with_suffix(".exit").write_text(f"{process.returncode}\n", encoding="utf-8")
    process.check_returncode()
    rows = []
    for case, operations, elapsed in csv.reader(process.stdout.splitlines()):
        operations, elapsed = int(operations), int(elapsed)
        assert MODES[mode][case] == operations and elapsed > 0
        assert mode != "jsonl" or peak > 0
        rows.append({
            "mode": mode,
            "case": case,
            "build": build,
            "ordinal": ordinal,
            "pair": None if warmup else (ordinal + 1) // 2,
            "warmup": warmup,
            "operations": operations,
            "elapsed_ns": elapsed,
            "peak_working_set_bytes": peak,
            "exit": process.returncode,
        })
    assert [r["case"] for r in rows] == list(MODES[mode])
    return rows


def summarize(rows, metric):
    base = [r[metric] for r in rows if r["build"] == "baseline"]
    trial = [r[metric] for r in rows if r["build"] == "trial"]
    ratios = [b / a for a, b in zip(base, trial, strict=True)]
    assert len(ratios) == 8
    ratio = statistics.median(ratios)
    faster = sum(r < 1 for r in ratios)
    return {
        "baseline_median": statistics.median(base),
        "trial_median": statistics.median(trial),
        "baseline_range": [min(base), max(base)],
        "trial_range": [min(trial), max(trial)],
        "median_paired_ratio": ratio,
        "faster_pairs": faster,
        "meets_margin": ratio <= 0.85 and faster >= 7,
    }


def main():
    identities = {label: digest(ROOT / f"{label}.exe") for label in ("baseline", "trial")}
    assert len(set(identities.values())) == 2
    fixture = ROOT / "fixture.jsonl"
    write_fixture(fixture)
    fixture_hash = digest(fixture)
    (ROOT / "runs").mkdir(exist_ok=True)
    rows = []
    with (ROOT / "samples.csv").open("w", encoding="utf-8", newline="") as file:
        writer = None
        for mode in MODES:
            schedule = [(0, "baseline", True), (0, "trial", True)]
            schedule.extend((i, build, False) for i, build in enumerate(ORDER, 1))
            for ordinal, build, warmup in schedule:
                sample = run_case(mode, build, ordinal, warmup, fixture)
                if writer is None:
                    writer = csv.DictWriter(file, fieldnames=list(sample[0]), lineterminator="\n")
                    writer.writeheader()
                writer.writerows(sample)
                file.flush()
                rows.extend(sample)

    results = {}
    for mode, cases in MODES.items():
        for case in cases:
            times = [r for r in rows if r["case"] == case and not r["warmup"]]
            results[case] = {"elapsed_ns": summarize(times, "elapsed_ns")}
            if mode == "jsonl":
                results[case]["peak_working_set_bytes"] = summarize(times, "peak_working_set_bytes")
    report = {
        "build_sha256": identities,
        "harness_sha256": digest(ROOT / "harness.rs"),
        "fixture_sha256": fixture_hash,
        "fixture_bytes": fixture.stat().st_size,
        "order": ORDER,
        "results": results,
    }
    (ROOT / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    for case, metrics in report["results"].items():
        for metric, result in metrics.items():
            unit = "MiB" if metric.startswith("peak") else "ms"
            scale = 2**20 if unit == "MiB" else 1e6
            print(
                f"{case:<19} {result['baseline_median'] / scale:8.3f} -> "
                f"{result['trial_median'] / scale:8.3f} {unit:<3} "
                f"{(1 - result['median_paired_ratio']) * 100:+5.1f}% paired reduction "
                f"({result['faster_pairs']}/8)"
            )


if __name__ == "__main__":
    main()
