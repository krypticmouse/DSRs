//! The load-bearing property of the structural checker: everything the full
//! parser accepts, [`dsrs_syntax::check`] accepts.
//!
//! Runs over local copies of the golden `.dsrs` fixtures maintained next to
//! the full parser (`crates/dspy-rs/tests/fixtures/*.dsrs`). Keep the copies
//! in sync when a fixture changes — `dspy-rs`'s `test_include_program.rs`
//! exercises the same artifacts through `include_program!`, so drift that
//! matters (a fixture the checker would reject) fails that suite too.

#[test]
fn parity_accepts_everything_the_full_parser_accepts() {
    let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let mut seen = 0usize;
    for entry in std::fs::read_dir(fixtures).expect("fixtures dir readable") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("dsrs") {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("fixture readable");
        dsrs_syntax::check(&src)
            .unwrap_or_else(|e| panic!("syntax checker rejected {}: {e}", path.display()));
        seen += 1;
    }
    assert!(seen >= 3, "expected the golden .dsrs fixtures, found {seen}");
}
