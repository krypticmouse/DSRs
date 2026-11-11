"""Compare two frozen rendering executables with identical fixed harnesses."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    harness = "crates/dspy-rs/examples/bench-input-rendering.rs"
    baseline_source = subprocess.check_output(["git", "show", f"{args.baseline_revision}:{harness}"], cwd=repo)
    candidate_source = subprocess.check_output(["git", "show", f"HEAD:{harness}"], cwd=repo)
    if baseline_source != candidate_source:
        parser.error("the baseline and candidate must use identical benchmark source")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        parser.error("commit the candidate before measuring")
    if args.out.resolve().is_relative_to(repo):
        parser.error("keep evidence outside the source checkout")
    args.out.mkdir(parents=True, exist_ok=False)
    binaries = {"baseline": args.baseline.resolve(strict=True), "candidate": args.candidate.resolve(strict=True)}
    manifest = {"baseline_revision": args.baseline_revision,
        "candidate_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(), "python": sys.version,
        "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
        "binaries": {name: digest(path) for name, path in binaries.items()},
        "harness_sha256_git_blob": hashlib.sha256(candidate_source).hexdigest(),
        "lock_sha256": digest(repo / "Cargo.lock"),
        "protocol": {"iterations": 2000, "warmup": 100, "samples_per_binary": 8,
            "order": "ABBA repeated four times", "allocation_iterations": 200,
            "acceptance": {"default_max_latency_ratio": 1.10, "jinja_8_16_max_latency_ratio": 0.90,
                "jinja_8_16_max_allocated_byte_ratio": 0.50}}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    allocations = {}
    for name, binary in binaries.items():
        dump = args.out / f"{name}-outputs"
        raw = subprocess.check_output([str(binary), "200", "--alloc", "--dump", str(dump)], text=True)
        (args.out / f"{name}-allocations.json").write_text(raw, encoding="utf-8")
        allocations[name] = {r["case"]: r for r in json.loads(raw)}
    hashes = {}
    for baseline in sorted((args.out / "baseline-outputs").glob("*.txt")):
        candidate = args.out / "candidate-outputs" / baseline.name
        assert baseline.read_bytes() == candidate.read_bytes(), baseline.name
        hashes[baseline.name] = digest(baseline)
    assert len(hashes) == 6
    (args.out / "output-sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    timings = {name: {} for name in binaries}
    for i, name in enumerate(["baseline", "candidate", "candidate", "baseline"] * 4):
        raw = subprocess.check_output([str(binaries[name]), "2000"], text=True)
        (args.out / f"timing-{i:02d}-{name}.json").write_text(raw, encoding="utf-8")
        for row in json.loads(raw):
            assert not row["allocation_profile"]
            assert row["output_hash"] == allocations[name][row["case"]]["output_hash"]
            timings[name].setdefault(row["case"], []).append(row["ns_per_op"])
        print(f"validated timing {i:02d} {name}", flush=True)
    summary = []
    for case in timings["baseline"]:
        before = statistics.median(timings["baseline"][case])
        after = statistics.median(timings["candidate"][case])
        byte_ratio = allocations["candidate"][case]["allocated_bytes_per_op"] / allocations["baseline"][case]["allocated_bytes_per_op"]
        latency_limit = 1.10 if case.startswith("default") else (0.90 if case in ["jinja-8", "jinja-16"] else None)
        accepted = after / before <= latency_limit if latency_limit is not None else None
        if case in ["jinja-8", "jinja-16"]:
            accepted = accepted and byte_ratio <= 0.50
        summary.append({"case": case, "baseline_ns_median": before, "candidate_ns_median": after,
            "latency_ratio": after / before, "allocated_byte_ratio": byte_ratio,
            "baseline_samples_ns": timings["baseline"][case], "candidate_samples_ns": timings["candidate"][case],
            "accepted": accepted})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
