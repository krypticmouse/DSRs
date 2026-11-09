# Fastant

A drop-in replacement for [`std::time::Instant`](https://doc.rust-lang.org/std/time/struct.Instant.html) that measures time with high performance and high accuracy powered by [Time Stamp Counter (TSC)](https://en.wikipedia.org/wiki/Time_Stamp_Counter).

[![Actions Status](https://github.com/fast/fastant/workflows/CI/badge.svg)](https://github.com/fast/fastant/actions)
[![Documentation](https://docs.rs/fastant/badge.svg)](https://docs.rs/fastant/)
[![Crates.io](https://img.shields.io/crates/v/fastant.svg)](https://crates.io/crates/fastant)
[![LICENSE](https://img.shields.io/github/license/fast/fastant.svg)](LICENSE)

## Usage

```toml
[dependencies]
fastant = "0.1"
```

```rust
fn main() {
    let start = fastant::Instant::now();
    let duration: std::time::Duration = start.elapsed();
}
```

## Motivation

This library is used by a high performance tracing library [`fastrace`](https://github.com/fast/fastrace). The main purpose is to use [Time Stamp Counter (TSC)](https://en.wikipedia.org/wiki/Time_Stamp_Counter) on x86 processors to measure time at high speed without losing much accuracy.

## Platform Support

Currently, only the Linux on `x86` or `x86_64` is backed by Time Stamp Counter (TSC). On other platforms, Fastant falls back to `std::time`. If TSC is unstable, it will also fall back to `std::time`.

If speed is privileged over accuracy when fallback occurs, you can use `fallback-coarse` feature to use coarse time:

```toml
[dependencies]
fastant = { version = "0.1", features = ["fallback-coarse"] }
```
