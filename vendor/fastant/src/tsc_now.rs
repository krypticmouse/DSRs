// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

//! This module will be compiled when it's either linux_x86 or linux_x86_64.

use std::cell::UnsafeCell;
use std::fs::read_to_string;
use std::io::ErrorKind;
use std::time::Instant;

static TSC_STATE: TSCState = TSCState {
    is_tsc_available: UnsafeCell::new(false),
    tsc_level: UnsafeCell::new(TSCLevel::Unstable),
    nanos_per_cycle: UnsafeCell::new(1.0),
};

struct TSCState {
    is_tsc_available: UnsafeCell<bool>,
    tsc_level: UnsafeCell<TSCLevel>,
    nanos_per_cycle: UnsafeCell<f64>,
}

unsafe impl Sync for TSCState {}

#[small_ctor::ctor]
unsafe fn init() {
    let tsc_level = TSCLevel::get();
    let is_tsc_available = match &tsc_level {
        TSCLevel::Stable { .. } => true,
        TSCLevel::Unstable => false,
    };
    if is_tsc_available {
        *TSC_STATE.nanos_per_cycle.get() = 1_000_000_000.0 / tsc_level.cycles_per_second() as f64;
    }
    *TSC_STATE.is_tsc_available.get() = is_tsc_available;
    *TSC_STATE.tsc_level.get() = tsc_level;
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
}

#[inline]
pub(crate) fn is_tsc_available() -> bool {
    unsafe { *TSC_STATE.is_tsc_available.get() }
}

#[inline]
pub(crate) fn nanos_per_cycle() -> f64 {
    unsafe { *TSC_STATE.nanos_per_cycle.get() }
}

#[inline]
pub(crate) fn current_cycle() -> u64 {
    match unsafe { &*TSC_STATE.tsc_level.get() } {
        TSCLevel::Stable {
            cycles_from_anchor, ..
        } => tsc().wrapping_sub(*cycles_from_anchor),
        TSCLevel::Unstable => panic!("tsc is unstable"),
    }
}

enum TSCLevel {
    Stable {
        cycles_per_second: u64,
        cycles_from_anchor: u64,
    },
    Unstable,
}

impl TSCLevel {
    fn get() -> TSCLevel {
        if !is_tsc_stable() {
            return TSCLevel::Unstable;
        }

        let anchor = Instant::now();
        let (cps, cfa) = cycles_per_sec(anchor);
        TSCLevel::Stable {
            cycles_per_second: cps,
            cycles_from_anchor: cfa,
        }
    }

    #[inline]
    fn cycles_per_second(&self) -> u64 {
        match self {
            TSCLevel::Stable {
                cycles_per_second, ..
            } => *cycles_per_second,
            TSCLevel::Unstable => panic!("tsc is unstable"),
        }
    }
}

/// If linux kernel detected TSCs are sync between CPUs, we can
/// rely on the result to say tsc is stable so that no need to
/// sync TSCs by ourselves.
fn is_tsc_stable() -> bool {
    has_invariant_tsc() || clock_source_has_tsc()
}

fn clock_source_has_tsc() -> bool {
    #[cfg(target_os = "linux")]
    {
        const CURRENT_CLOCKSOURCE: &str =
            "/sys/devices/system/clocksource/clocksource0/current_clocksource";
        const AVAILABLE_CLOCKSOURCE: &str =
            "/sys/devices/system/clocksource/clocksource0/available_clocksource";

        match read_to_string(CURRENT_CLOCKSOURCE) {
            Ok(content) => content.contains("tsc"),
            Err(e) if e.kind() == ErrorKind::NotFound => {
                // we only check `available_clocksource` iff `current_clocksource` not exists.
                read_to_string(AVAILABLE_CLOCKSOURCE)
                    .map(|s| s.contains("tsc"))
                    .unwrap_or(false)
            }
            Err(_) => false,
        }
    }

    #[cfg(not(target_os = "linux"))]
    false
}

/// Invariant TSC could make sure TSC got synced among multi CPUs.
/// They will be reset at same time, and run in same frequency.
/// But in some VM, the max Extended Function in CPUID is < 0x80000007,
/// we should enable TSC if the system clock source is TSC.
#[inline]
fn has_invariant_tsc() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        use core::arch::x86_64::__cpuid;
        let cpuid_invariant_tsc_bts = 1 << 8;
        __cpuid(0x80000000).eax >= 0x80000007
            && __cpuid(0x80000007).edx & cpuid_invariant_tsc_bts != 0
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    false
}

/// Returns (1) cycles per second and (2) cycles from anchor.
/// The result of subtracting `cycles_from_anchor` from newly fetched TSC
/// can be used to
///   1. readjust TSC to begin from zero
///   2. sync TSCs between all CPUs
fn cycles_per_sec(anchor: Instant) -> (u64, u64) {
    let (cps, last_monotonic, last_tsc) = _cycles_per_sec();
    let nanos_from_anchor = (last_monotonic - anchor).as_nanos();
    let cycles_flied = cps as f64 * nanos_from_anchor as f64 / 1_000_000_000.0;
    let cycles_from_anchor = last_tsc - cycles_flied.ceil() as u64;

    (cps, cycles_from_anchor)
}

/// Returns (1) cycles per second, (2) last monotonic time and (3) associated tsc.
fn _cycles_per_sec() -> (u64, Instant, u64) {
    let mut cycles_per_sec;
    let mut last_monotonic;
    let mut last_tsc;
    let mut old_cycles = 0.0;

    loop {
        let (t1, tsc1) = monotonic_with_tsc();
        loop {
            let (t2, tsc2) = monotonic_with_tsc();
            last_monotonic = t2;
            last_tsc = tsc2;
            let elapsed_nanos = (t2 - t1).as_nanos();
            if elapsed_nanos > 10_000_000 {
                cycles_per_sec = (tsc2 - tsc1) as f64 * 1_000_000_000.0 / elapsed_nanos as f64;
                break;
            }
        }
        let delta = f64::abs(cycles_per_sec - old_cycles);
        if delta / cycles_per_sec < 0.00001 {
            break;
        }
        old_cycles = cycles_per_sec;
    }

    (cycles_per_sec.round() as u64, last_monotonic, last_tsc)
}

/// Try to get tsc and monotonic time at the same time. Due to
/// get interrupted in half way may happen, they aren't guaranteed
/// to represent the same instant.
fn monotonic_with_tsc() -> (Instant, u64) {
    (Instant::now(), tsc())
}

#[inline]
fn tsc() -> u64 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::_rdtsc;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::_rdtsc;

    unsafe { _rdtsc() }
}
