//! IR-7 (RFC 0002 §6.1): `include_program!` through the `dspy_rs` re-export —
//! the path users actually write (`dspy_rs::include_program!`). The macro
//! resolves the runtime crate via proc-macro-crate, so this exercises the
//! `FoundCrate::Itself` → `::dspy_rs` alias branch that examples/tests of the
//! dspy-rs package itself hit.

dspy_rs::include_program!("tests/fixtures/qa.dsrs");

#[test]
fn embedded_program_agrees_with_the_direct_loader() {
    let embedded = qa::program();
    let loaded = dspy_rs::ir::Program::load_dsrs(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qa.dsrs"
    ))
    .expect("golden fixture loads");
    assert_eq!(embedded.meta.name, loaded.meta.name);
    assert_eq!(
        embedded.meta.program_hash, loaded.meta.program_hash,
        "embed and load agree on the content hash"
    );
    assert_eq!(embedded.to_dsrs(), loaded.to_dsrs());
}
