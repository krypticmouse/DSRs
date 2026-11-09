// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

//! A drop-in replacement for [`std::time::Instant`](https://doc.rust-lang.org/std/time/struct.Instant.html)
//! that measures time with high performance and high accuracy powered by [Time Stamp Counter (TSC)](https://en.wikipedia.org/wiki/Time_Stamp_Counter).
//!
//! ## Example
//!
//! ```rust
//! let start = fastant::Instant::now();
//! let duration: std::time::Duration = start.elapsed();
//! ```
//!
//! ## Platform Support
//!
//! Currently, only the Linux on `x86` or `x86_64` is backed by Time Stamp Counter (TSC).
//! On other platforms, `fastant` falls back to coarse time.
//!
//! ## Calibration
//!
//! Time Stamp Counter (TSC) doesn't necessarily tick in constant speed and even doesn't synchronize
//! across CPU cores. The calibration detects the TSC deviation and calculates the correction
//! factors with the assistance of a source wall clock. Once the deviation is beyond a crazy
//! threshold, the calibration will fail, and then we will fall back to coarse time.
//!
//! This calibration is stored globally and reused. In order to start the calibration before any
//! call to `fastant` as to make sure that the time spent on `fastant` is constant, we link the
//! calibration into application's initialization linker section, so it'll get executed once the
//! process starts.
//!
//! **[See also the `Instant` type](Instant).**

#![cfg_attr(docsrs, feature(doc_cfg))]

mod instant;
// Miri cannot execute the CPUID inline assembly in the TSC startup constructor.
// Use the same clock fallback as other platforms while interpreting tests.
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86", target_arch = "x86_64"),
    not(miri)
))]
mod tsc_now;

pub use instant::Anchor;
#[cfg(all(feature = "atomic", target_has_atomic = "64"))]
#[cfg_attr(docsrs, doc(cfg(all(feature = "atomic", target_has_atomic = "64"))))]
pub use instant::Atomic;
pub use instant::Instant;

/// Return `true` if the current platform supports Time Stamp Counter (TSC),
/// and the calibration has succeeded.
///
/// The result is always the same during the lifetime of the application process.
#[inline]
pub fn is_tsc_available() -> bool {
    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86", target_arch = "x86_64"),
        not(miri)
    ))]
    {
        tsc_now::is_tsc_available()
    }
    #[cfg(not(all(
        target_os = "linux",
        any(target_arch = "x86", target_arch = "x86_64"),
        not(miri)
    )))]
    {
        false
    }
}

#[inline]
pub(crate) fn current_cycle() -> u64 {
    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86", target_arch = "x86_64"),
        not(miri)
    ))]
    {
        if tsc_now::is_tsc_available() {
            tsc_now::current_cycle()
        } else {
            current_cycle_fallback()
        }
    }
    #[cfg(not(all(
        target_os = "linux",
        any(target_arch = "x86", target_arch = "x86_64"),
        not(miri)
    )))]
    {
        current_cycle_fallback()
    }
}

#[cfg(not(feature = "fallback-coarse"))]
pub(crate) fn current_cycle_fallback() -> u64 {
    web_time::SystemTime::now()
        .duration_since(web_time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

#[cfg(feature = "fallback-coarse")]
pub(crate) fn current_cycle_fallback() -> u64 {
    let coarse = coarsetime::Instant::now_without_cache_update();
    coarsetime::Duration::from_ticks(coarse.as_ticks()).as_nanos()
}

#[inline]
pub(crate) fn nanos_per_cycle() -> f64 {
    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86", target_arch = "x86_64"),
        not(miri)
    ))]
    {
        tsc_now::nanos_per_cycle()
    }
    #[cfg(not(all(
        target_os = "linux",
        any(target_arch = "x86", target_arch = "x86_64"),
        not(miri)
    )))]
    {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;
    use std::time::Instant as StdInstant;

    use rand::Rng;
    use wasm_bindgen_test::wasm_bindgen_test;

    use super::*;

    #[test]
    #[wasm_bindgen_test]
    fn test_is_tsc_available() {
        let _ = is_tsc_available();
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_monotonic() {
        let mut prev = 0;
        for _ in 0..10000 {
            let cur = current_cycle();
            assert!(cur >= prev);
            prev = cur;
        }
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_nanos_per_cycle() {
        let _ = nanos_per_cycle();
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_unix_time() {
        let now = Instant::now();
        let anchor = Anchor::new();
        let unix_nanos = now.as_unix_nanos(&anchor);
        assert!(unix_nanos > 0);
    }

    #[test]
    fn test_duration() {
        let mut rng = rand::rng();
        for _ in 0..10 {
            let instant = Instant::now();
            let std_instant = StdInstant::now();
            std::thread::sleep(Duration::from_millis(rng.random_range(100..500)));
            let check = move || {
                let duration_ns_fastant = instant.elapsed();
                let duration_ns_std = std_instant.elapsed();

                #[cfg(target_os = "windows")]
                let expect_max_delta_ns = 40_000_000;
                #[cfg(not(target_os = "windows"))]
                let expect_max_delta_ns = 5_000_000;

                let real_delta = (duration_ns_std.as_nanos() as i128
                    - duration_ns_fastant.as_nanos() as i128)
                    .abs();
                assert!(
                    real_delta < expect_max_delta_ns,
                    "real delta: {}",
                    real_delta
                );
            };
            check();
            std::thread::spawn(check)
                .join()
                .expect("failed to join thread");
        }
    }
}
