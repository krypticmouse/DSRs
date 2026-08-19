//! Stable 64-bit hashing for trace request hashes and LM cache keys.
//!
//! `std::hash::DefaultHasher` is not guaranteed stable across Rust releases;
//! replay fixtures and on-disk trace files must survive toolchain upgrades, so
//! everything that hashes for identity uses FNV-1a with a fixed algorithm.

use std::hash::Hasher;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a 64-bit hasher: deterministic across platforms and Rust versions.
#[derive(Clone, Debug)]
pub struct StableHasher(u64);

impl StableHasher {
    pub fn new() -> Self {
        Self(FNV_OFFSET)
    }
}

impl Default for StableHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher for StableHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(FNV_PRIME);
        }
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

/// Adapts a [`Hasher`] into a [`std::fmt::Write`] sink so values can be hashed
/// through their `Debug`/`Display` representation without materializing a string.
pub struct HashWriter<'a, H: Hasher>(pub &'a mut H);

impl<H: Hasher> std::fmt::Write for HashWriter<'_, H> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.write(s.as_bytes());
        Ok(())
    }
}

/// Hashes a `Debug`-formatted value with the stable hasher.
pub fn stable_hash_debug<T: std::fmt::Debug>(value: &T) -> u64 {
    use std::fmt::Write as _;
    let mut hasher = StableHasher::new();
    let _ = write!(HashWriter(&mut hasher), "{value:?}");
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_matches_reference_vectors() {
        // Standard FNV-1a 64 test vectors.
        let mut h = StableHasher::new();
        h.write(b"");
        assert_eq!(h.finish(), 0xcbf2_9ce4_8422_2325);

        let mut h = StableHasher::new();
        h.write(b"a");
        assert_eq!(h.finish(), 0xaf63_dc4c_8601_ec8c);

        let mut h = StableHasher::new();
        h.write(b"foobar");
        assert_eq!(h.finish(), 0x85944171f73967e8);
    }

    #[test]
    fn debug_hash_is_deterministic() {
        assert_eq!(
            stable_hash_debug(&("abc", 42)),
            stable_hash_debug(&("abc", 42))
        );
        assert_ne!(
            stable_hash_debug(&("abc", 42)),
            stable_hash_debug(&("abc", 43))
        );
    }
}
