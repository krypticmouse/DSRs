"""Paired steady-state system-prompt benchmark; retain all samples and bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    git = lambda *cmd: subprocess.check_output(["git", *cmd], cwd=repo)
    if git("status", "--porcelain").strip():
        parser.error("commit the candidate before measuring")
    candidate = git("rev-parse", "HEAD").decode().strip()
    baseline = git("rev-parse", args.baseline_revision).decode().strip()
    harness = "crates/dspy-rs/examples/bench-system-prompts.rs"
    source = git("show", f"{candidate}:{harness}")
    assert source == git("show", f"{baseline}:{harness}"), "different workloads"
    args.out.mkdir(parents=True, exist_ok=False)
    executables = {"baseline": args.baseline.resolve(strict=True), "candidate": args.candidate.resolve(strict=True)}
    manifest = {"baseline": baseline, "candidate": candidate,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(), "python": sys.version,
        "rustc": subprocess.check_output(["rustc", "+1.98.1", "-vV"], text=True),
        "harness_sha256": sha(source), "lock_sha256": sha((repo / "Cargo.lock").read_bytes()),
        "executables": {name: sha(exe.read_bytes()) for name, exe in executables.items()},
        "protocol": {"iterations": 20000, "warmup": 100, "samples_per_binary": 8,
            "order": "ABBA repeated four times", "max_median_latency_ratio": 0.90}}
    assert git("show", f"{baseline}:Cargo.lock") == git("show", f"{candidate}:Cargo.lock")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name, exe in executables.items():
        subprocess.run([str(exe), "1", "--dump", str(args.out / name)], check=True, capture_output=True)
    checksums = {}
    for width in [1, 8, 16]:
        name = f"system-{width}.json"
        expected = (args.out / "baseline" / name).read_bytes()
        assert expected == (args.out / "candidate" / name).read_bytes(), name
        checksums[name] = sha(expected)
    (args.out / "output-sha256.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    samples = {name: {} for name in executables}
    for index, name in enumerate(["baseline", "candidate", "candidate", "baseline"] * 4):
        result = subprocess.run([str(executables[name]), "20000"], text=True, capture_output=True)
        (args.out / f"timing-{index:02d}-{name}.json").write_text(result.stdout, encoding="utf-8")
        (args.out / f"timing-{index:02d}-{name}.stderr.txt").write_text(result.stderr, encoding="utf-8")
        result.check_returncode()
        for row in json.loads(result.stdout):
            assert row["iterations"] == 20000 and row["warmup"] == 100
            samples[name].setdefault(row["case"], []).append(row["ns_per_op"])
    summary = []
    for case, values in samples["baseline"].items():
        candidate_values = samples["candidate"][case]
        ratio = statistics.median(candidate_values) / statistics.median(values)
        summary.append({"case": case, "baseline_samples_ns": values,
            "candidate_samples_ns": candidate_values, "latency_ratio": ratio, "accepted": ratio <= 0.90})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not all(row["accepted"] for row in summary):
        raise SystemExit("predeclared performance gate failed")


if __name__ == "__main__":
    main()
