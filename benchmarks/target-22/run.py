"""Assess all three #22 baselines without treating a single workload as universal."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import urllib.request


def summarize(records, modes, frameworks):
    summary = []
    for mode in modes:
        samples = {framework: [r['mean_ns'] for r in records if r.get('valid') and (r['mode'], r['framework']) == (mode, framework)] for framework in frameworks}
        valid = all(len(values) == 8 for values in samples.values())
        speedup = statistics.median(samples['dspy']) / statistics.median(samples['dsrs']) if valid else None
        summary.append({'mode': mode, 'valid': valid, 'run_mean_ns': samples,
            'speedup': speedup, 'baseline_speedup': statistics.median(samples['baseline']) / statistics.median(samples['dsrs']) if valid else None,
            'meets_2x': speedup >= 2 if valid else None})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-record", type=Path, required=True)
    parser.add_argument("--candidate-record", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--ollama-log", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modes", nargs='+', choices=['live', 'cached', 'offline'], default=['live', 'cached', 'offline'])
    parser.add_argument("--idle-timeout", type=int, default=1800)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    git = lambda *cmd: subprocess.check_output(["git", *cmd], cwd=repo, text=True).strip()
    assert not git("status", "--porcelain"), "freeze the source first"
    baseline_record = json.loads(args.baseline_record.read_text(encoding='utf-8-sig'))
    candidate_record = json.loads(args.candidate_record.read_text(encoding='utf-8-sig'))
    assert baseline_record['rust_binary_sha256'] == sha(args.baseline)
    assert candidate_record['rust_binary_sha256'] == sha(args.rust)
    assert baseline_record['rustc'] == candidate_record['rustc']
    assert baseline_record['command'] == candidate_record['command']
    assert baseline_record['lock_sha256'] == candidate_record['lock_sha256'] == sha(repo / 'Cargo.lock')
    workload = 'crates/dspy-rs/examples/bench-target-22.rs'
    assert baseline_record['harness_sha256'][workload] == sha(repo / workload), 'Rust workload changed since baseline build'
    assert candidate_record['harness_sha256'][workload] == sha(repo / workload), 'Rust workload changed since candidate build'
    args.out.mkdir(parents=True, exist_ok=False)
    def compiler_jobs():
        command = "Get-CimInstance Win32_Process -Filter \"Name = 'cargo.exe' OR Name = 'rustc.exe'\" | Select-Object ProcessId,Name | ConvertTo-Json -Compress"
        result = subprocess.check_output(['powershell', '-NoProfile', '-Command', command], text=True).strip()
        return json.loads(result) if result else []
    deadline = time.monotonic() + args.idle_timeout
    def wait_for_idle():
        while compiler_jobs():
            if time.monotonic() > deadline:
                raise TimeoutError('no compiler-idle window before deadline')
            print('waiting for compiler-idle window', flush=True)
            time.sleep(5)
    wait_for_idle()
    def count_requests():
        log = subprocess.check_output(["wsl", "-d", "Ubuntu-24.04", "--", "cat", args.ollama_log], text=True)
        return sum('POST' in line and '/v1/chat/completions' in line for line in log.splitlines())
    def api(path):
        with urllib.request.urlopen(args.base_url.removesuffix('/v1') + path, timeout=10) as response:
            return json.load(response)
    manifest = {"revision": candidate_record['revision'], "controller_revision": git("rev-parse", "HEAD"), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(), "driver_python": sys.version,
        "model": api('/api/tags'), "ollama": api('/api/version'),
        "rustc": subprocess.check_output(["rustc", "+1.98.1", "-vV"], text=True),
        "framework_python": subprocess.check_output([str(args.python), "-V"], text=True).strip(),
        "python_packages": json.loads(subprocess.check_output([str(args.python), "-c",
            "import importlib.metadata,json; print(json.dumps(dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()))))"], text=True)),
        "rust_binary_sha256": sha(args.rust), "lock_sha256": sha(repo / 'Cargo.lock'), "modes": args.modes,
        "baseline": {"revision": baseline_record['revision'], "binary_sha256": sha(args.baseline),
            "record_sha256": sha(args.baseline_record), "rustc": baseline_record['rustc']},
        "harness_sha256": {name: sha(repo / name) for name in [
            'crates/dspy-rs/examples/bench-target-22.rs', 'benchmarks/target-22/dspy_case.py', 'benchmarks/target-22/run.py']},
        "protocol": {"order": "baseline-candidate-DSPy-DSPy-candidate-baseline repeated four times per mode", "samples_per_framework": 8,
            "iterations": {"live": 1, "cached": 1000, "offline": 1000},
            "warmup": {"live": 1, "cached": 100, "offline": 100},
            "temperature": 0, "max_tokens": 64, "retries": 0, "quality": "answer exactly Paris", "target_speedup": 2,
            "output_format": "native JSON Schema, explicit custom endpoint opt-in", "history": "DSPy disabled; DSRs capacities zero"}}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    commands = {'baseline': [str(args.baseline.resolve())], 'dsrs': [str(args.rust.resolve())], 'dspy': [str(args.python.resolve()), str(repo / 'benchmarks/target-22/dspy_case.py')]}
    records = []
    def persist():
        (args.out / 'requests-and-means.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
        (args.out / 'summary.json').write_text(json.dumps(summarize(records, args.modes, commands), indent=2), encoding='utf-8')
    for mode in args.modes:
        for index, framework in enumerate(['baseline', 'dsrs', 'dspy', 'dspy', 'dsrs', 'baseline'] * 4):
            attempt = 0
            while True:
                wait_for_idle()
                before = count_requests()
                jobs_before = compiler_jobs()
                result = subprocess.run(commands[framework] + [mode], cwd=repo, text=True, capture_output=True, timeout=180)
                jobs_after = compiler_jobs()
                after = count_requests()
                stem = args.out / f'{mode}-{index:02d}-{framework}-attempt-{attempt:02d}'
                stem.with_suffix('.json').write_text(result.stdout, encoding='utf-8')
                stem.with_suffix('.stderr.txt').write_text(result.stderr, encoding='utf-8')
                stem.with_suffix('.background.json').write_text(json.dumps({'before': jobs_before, 'after': jobs_after}, indent=2), encoding='utf-8')
                record = {'mode': mode, 'framework': framework, 'index': index, 'attempt': attempt,
                    'artifact': stem.name, 'exit_code': result.returncode, 'http_requests': after - before}
                if jobs_before or jobs_after:
                    records.append(dict(record, valid=False, reason='compiler activity'))
                    persist()
                    print(f'excluded {stem.name}: compiler activity; retrying same position', flush=True)
                    attempt += 1
                    if time.monotonic() > deadline:
                        raise TimeoutError('compiler activity interrupted collection until deadline')
                    continue
                if result.returncode != 0:
                    records.append(dict(record, valid=False, reason='correctness command failed'))
                    persist()
                    raise RuntimeError(f'{stem.name} failed its correctness gate')
                data = json.loads(result.stdout)
                assert data['answer'] == 'Paris' and len(data['samples_ns']) == data['iterations']
                expected = {'live': 2, 'cached': 1, 'offline': 0}[mode]
                assert after - before == expected, (framework, mode, after - before, expected)
                records.append(dict(record, valid=True, mean_ns=statistics.mean(data['samples_ns'])))
                persist()
                print(f'verified {stem.name}', flush=True)
                break
    print(json.dumps(summarize(records, args.modes, commands), indent=2))


if __name__ == '__main__':
    main()
