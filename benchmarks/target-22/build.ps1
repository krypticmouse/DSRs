param(
    [Parameter(Mandatory)][string]$Worktree,
    [Parameter(Mandatory)][string]$TargetDir,
    [Parameter(Mandatory)][string]$OutputDir,
    [Parameter(Mandatory)][ValidateSet('baseline', 'candidate')][string]$Kind
)
$ErrorActionPreference = 'Stop'
$buildTree = (Resolve-Path -LiteralPath $Worktree).Path
$targetPath = [IO.Path]::GetFullPath($TargetDir)
$outputPath = [IO.Path]::GetFullPath($OutputDir)
if (Get-Process cargo, rustc -ErrorAction SilentlyContinue) {
    throw 'Serialize Cargo work before cleaning or building this target.'
}
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
Set-Location -LiteralPath $buildTree
if (@(& git status --porcelain).Count -ne 0) { throw 'Freeze the benchmark source first.' }
$revision = & git rev-parse HEAD
$env:CARGO_TARGET_DIR = $targetPath
$env:CARGO_INCREMENTAL = '0'
& cargo +1.98.1 clean -p dspy-rs --profile release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$arguments = @('+1.98.1', 'build', '-p', 'dspy-rs', '--release', '--no-default-features', '--features', 'cache', '--locked', '--example', 'bench-target-22', '-j1')
$started = [DateTime]::UtcNow.ToString('o')
$p = Start-Process cargo -ArgumentList $arguments -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput "$outputPath/$Kind.stdout.txt" -RedirectStandardError "$outputPath/$Kind.stderr.txt"
if ($p.ExitCode -ne 0) { exit $p.ExitCode }
Copy-Item -LiteralPath "$targetPath/release/examples/bench-target-22.exe" -Destination "$outputPath/$Kind.exe"
$workload = 'crates/dspy-rs/examples/bench-target-22.rs'
@{
    revision = $revision; command = (@('cargo') + $arguments)
    started = $started; ended = [DateTime]::UtcNow.ToString('o'); exit_code = $p.ExitCode
    rustc = @(& rustc +1.98.1 -vV)
    rust_binary_sha256 = (Get-FileHash -LiteralPath "$outputPath/$Kind.exe" -Algorithm SHA256).Hash.ToLowerInvariant()
    lock_sha256 = (Get-FileHash -LiteralPath Cargo.lock -Algorithm SHA256).Hash.ToLowerInvariant()
    harness_sha256 = @{$workload = (Get-FileHash -LiteralPath $workload -Algorithm SHA256).Hash.ToLowerInvariant()}
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$outputPath/$Kind.json"
