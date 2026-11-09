// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Sub;
use std::ops::SubAssign;
use std::time::Duration;

use web_time::SystemTime;
use web_time::UNIX_EPOCH;

/// A measurement of a monotonically non-decreasing clock. Similar to
/// [`std::time::Instant`](std::time::Instant) but is faster and more
/// accurate if TSC is available.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Instant(u64);

impl Instant {
    /// A default `Instant` that can be seen as a fixed but random moment.
    pub const ZERO: Instant = Instant(0);

    #[inline]
    /// Returns an instant corresponding to "now".
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fastant::Instant;
    /// let now = Instant::now();
    /// ```
    pub fn now() -> Instant {
        Instant(crate::current_cycle())
    }

    /// Returns the amount of time elapsed from another instant to this one,
    /// or zero duration if that instant is later than this one.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::thread::sleep;
    /// use std::time::Duration;
    ///
    /// use fastant::Instant;
    ///
    /// let now = Instant::now();
    /// sleep(Duration::new(1, 0));
    ///
    /// let new_now = Instant::now();
    /// println!("{:?}", new_now.duration_since(now));
    /// println!("{:?}", now.duration_since(new_now)); // 0ns
    /// ```
    pub fn duration_since(&self, earlier: Instant) -> Duration {
        self.checked_duration_since(earlier).unwrap_or_default()
    }

    /// Returns the amount of time elapsed from another instant to this one,
    /// or None if that instant is later than this one.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::thread::sleep;
    /// use std::time::Duration;
    ///
    /// use fastant::Instant;
    ///
    /// let now = Instant::now();
    /// sleep(Duration::new(1, 0));
    ///
    /// let new_now = Instant::now();
    /// println!("{:?}", new_now.checked_duration_since(now));
    /// println!("{:?}", now.checked_duration_since(new_now)); // None
    /// ```
    pub fn checked_duration_since(&self, earlier: Instant) -> Option<Duration> {
        Some(Duration::from_nanos(
            (self.0.checked_sub(earlier.0)? as f64 * crate::nanos_per_cycle()) as u64,
        ))
    }

    /// Returns the amount of time elapsed from another instant to this one,
    /// or zero duration if that instant is later than this one.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::thread::sleep;
    /// use std::time::Duration;
    ///
    /// use fastant::Instant;
    ///
    /// let now = Instant::now();
    /// sleep(Duration::new(1, 0));
    ///
    /// let new_now = Instant::now();
    /// println!("{:?}", new_now.saturating_duration_since(now));
    /// println!("{:?}", now.saturating_duration_since(new_now)); // 0ns
    /// ```
    pub fn saturating_duration_since(&self, earlier: Instant) -> Duration {
        self.checked_duration_since(earlier).unwrap_or_default()
    }

    /// Returns the amount of time elapsed since this instant was created.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::thread::sleep;
    /// use std::time::Duration;
    ///
    /// use fastant::Instant;
    ///
    /// let instant = Instant::now();
    /// let three_secs = Duration::from_secs(3);
    /// sleep(three_secs);
    /// assert!(instant.elapsed() >= three_secs);
    /// ```
    #[inline]
    pub fn elapsed(&self) -> Duration {
        Instant::now() - *self
    }

    /// Returns `Some(t)` where `t` is the time `self + duration` if `t` can be represented as
    /// `Instant` (which means it's inside the bounds of the underlying data structure), `None`
    /// otherwise.
    pub fn checked_add(&self, duration: Duration) -> Option<Instant> {
        self.0
            .checked_add((duration.as_nanos() as u64 as f64 / crate::nanos_per_cycle()) as u64)
            .map(Instant)
    }

    /// Returns `Some(t)` where `t` is the time `self - duration` if `t` can be represented as
    /// `Instant` (which means it's inside the bounds of the underlying data structure), `None`
    /// otherwise.
    pub fn checked_sub(&self, duration: Duration) -> Option<Instant> {
        self.0
            .checked_sub((duration.as_nanos() as u64 as f64 / crate::nanos_per_cycle()) as u64)
            .map(Instant)
    }

    /// Convert internal clocking counter into a UNIX timestamp represented as the
    /// nanoseconds elapsed from [UNIX_EPOCH](UNIX_EPOCH).
    ///
    /// [`Anchor`](Anchor) contains the necessary calibration data for conversion.
    /// Typically, initializing an [`Anchor`](Anchor) takes about 50 nanoseconds, so
    /// try to reuse it for a batch of `Instant`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::UNIX_EPOCH;
    ///
    /// use fastant::Anchor;
    /// use fastant::Instant;
    ///
    /// let anchor = Anchor::new();
    /// let instant = Instant::now();
    ///
    /// let expected = UNIX_EPOCH.elapsed().unwrap().as_nanos();
    /// assert!((instant.as_unix_nanos(&anchor) as i64 - expected as i64).abs() < 1_000_000);
    /// ```
    pub fn as_unix_nanos(&self, anchor: &Anchor) -> u64 {
        if self.0 > anchor.cycle {
            let forward_ns = ((self.0 - anchor.cycle) as f64 * crate::nanos_per_cycle()) as u64;
            anchor.unix_time_ns + forward_ns
        } else {
            let backward_ns = ((anchor.cycle - self.0) as f64 * crate::nanos_per_cycle()) as u64;
            anchor.unix_time_ns - backward_ns
        }
    }
}

impl Add<Duration> for Instant {
    type Output = Instant;

    fn add(self, other: Duration) -> Instant {
        self.checked_add(other)
            .expect("overflow when adding duration to instant")
    }
}

impl AddAssign<Duration> for Instant {
    fn add_assign(&mut self, other: Duration) {
        *self = *self + other;
    }
}

impl Sub<Duration> for Instant {
    type Output = Instant;

    fn sub(self, other: Duration) -> Instant {
        self.checked_sub(other)
            .expect("overflow when subtracting duration from instant")
    }
}

impl SubAssign<Duration> for Instant {
    fn sub_assign(&mut self, other: Duration) {
        *self = *self - other;
    }
}

impl Sub<Instant> for Instant {
    type Output = Duration;

    /// Returns the amount of time elapsed from another instant to this one,
    /// or zero duration if that instant is later than this one.
    fn sub(self, other: Instant) -> Duration {
        self.duration_since(other)
    }
}

impl std::fmt::Debug for Instant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// An anchor which can be used to convert internal clocking counter into system timestamp.
///
/// **[See also the `Instant::as_unix_nanos()`](Instant::as_unix_nanos).**
#[derive(Copy, Clone)]
pub struct Anchor {
    unix_time_ns: u64,
    cycle: u64,
}

impl Default for Anchor {
    fn default() -> Self {
        Self::new()
    }
}

impl Anchor {
    #[inline]
    pub fn new() -> Anchor {
        let unix_time_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("unexpected time drift")
            .as_nanos() as u64;
        Anchor {
            unix_time_ns,
            cycle: crate::current_cycle(),
        }
    }
}

#[cfg(all(feature = "atomic", target_has_atomic = "64"))]
#[cfg_attr(docsrs, doc(cfg(all(feature = "atomic", target_has_atomic = "64"))))]
mod atomic {
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering;

    use super::Instant;

    /// Atomic variant of [`Instant`].
    #[derive(Debug)]
    #[repr(transparent)]
    pub struct Atomic(AtomicU64);

    impl Atomic {
        /// Maximum with the current value.
        ///
        /// Finds the maximum of the current value and the argument `val`, and
        /// sets the new value to the result.
        ///
        /// Returns the previous value.
        ///
        /// `fetch_max` takes an [`Ordering`] argument which describes the memory ordering
        /// of this operation. All ordering modes are possible. Note that using
        /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
        /// using [`Release`] makes the load part [`Relaxed`].
        ///
        /// **Note**: This method is only available on platforms that support atomic operations on
        /// `[u64]`.
        #[inline]
        pub fn fetch_max(&self, val: Instant, order: Ordering) -> Instant {
            Instant(self.0.fetch_max(val.0, order))
        }

        /// Minimum with the current value.
        ///
        /// Finds the minimum of the current value and the argument `val`, and
        /// sets the new value to the result.
        ///
        /// Returns the previous value.
        ///
        /// `fetch_min` takes an [`Ordering`] argument which describes the memory ordering
        /// of this operation. All ordering modes are possible. Note that using
        /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
        /// using [`Release`] makes the load part [`Relaxed`].
        ///
        /// **Note**: This method is only available on platforms that support atomic operations on
        /// `[u64]`.
        #[inline]
        pub fn fetch_min(&self, val: Instant, order: Ordering) -> Instant {
            Instant(self.0.fetch_min(val.0, order))
        }

        /// Consumes the atomic and returns the contained [`Instant`].
        ///
        /// This is safe because passing `self` by value guarantees that no other threads are
        /// concurrently accessing the atomic data.
        #[inline]
        pub fn into_instant(self) -> Instant {
            Instant(self.0.into_inner())
        }

        /// Loads a value from the [`Atomic`].
        ///
        /// `load` takes an [`Ordering`] argument which describes the memory ordering of this
        /// operation. Possible values are [`SeqCst`], [`Acquire`] and [`Relaxed`].
        ///
        /// # Panics
        ///
        /// Panics if `order` is [`Release`] or [`AcqRel`].
        #[inline]
        pub fn load(&self, order: Ordering) -> Instant {
            Instant(self.0.load(order))
        }

        /// Creates a new [`Atomic`].
        #[inline]
        pub fn new(v: Instant) -> Self {
            Self(AtomicU64::new(v.0))
        }

        /// Stores a value into the [`Atomic`].
        ///
        /// `store` takes an [`Ordering`] argument which describes the memory ordering of this
        /// operation.  Possible values are [`SeqCst`], [`Release`] and [`Relaxed`].
        ///
        /// # Panics
        ///
        /// Panics if `order` is [`Acquire`] or [`AcqRel`].
        #[inline]
        pub fn store(&self, val: Instant, order: Ordering) {
            self.0.store(val.0, order)
        }

        /// Stores a value into the [`Atomic`], returning the previous value.
        ///
        /// `swap` takes an [`Ordering`] argument which describes the memory ordering
        /// of this operation. All ordering modes are possible. Note that using
        /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
        /// using [`Release`] makes the load part [`Relaxed`].
        ///
        /// **Note**: This method is only available on platforms that support atomic operations on
        /// `u64`
        #[inline]
        pub fn swap(&self, val: Instant, order: Ordering) -> Instant {
            Instant(self.0.swap(val.0, order))
        }
    }

    impl From<Instant> for Atomic {
        #[inline]
        fn from(instant: Instant) -> Self {
            Self::new(instant)
        }
    }
}

#[cfg(all(feature = "atomic", target_has_atomic = "64"))]
#[cfg_attr(docsrs, doc(cfg(all(feature = "atomic", target_has_atomic = "64"))))]
pub use atomic::Atomic;
