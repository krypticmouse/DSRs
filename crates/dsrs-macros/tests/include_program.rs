//! `include_program!` happy path (RFC 0002 §6.1): a valid `.dsrs` artifact
//! compiles, embeds its source, and fully parses/validates at first use.
//! The macro also generates a hidden `#[cfg(test)]` validation test inside
//! the emitted module — it runs as part of this test binary.

dsrs_macros::include_program!("tests/programs/qa.dsrs");

#[test]
fn embedded_program_loads_and_validates() {
    let program = qa::program();
    assert_eq!(&*program.meta.name, "qa");
    assert_eq!(program.meta.format, 1);
    assert!(program.meta.program_hash != 0);

    // The embedded text is the artifact byte-for-byte.
    let on_disk = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/programs/qa.dsrs"
    ))
    .expect("fixture readable");
    assert_eq!(qa::SOURCE, on_disk);

    // Full parse agrees with the direct API on the content hash.
    let direct = dspy_rs::ir::Program::from_dsrs(qa::SOURCE).expect("fixture parses");
    assert_eq!(direct.meta.program_hash, program.meta.program_hash);
}

#[test]
fn try_program_is_ok_and_cached() {
    let a = qa::try_program().expect("valid artifact");
    let b = qa::program();
    assert!(std::ptr::eq(a, b), "LazyLock caches one Program value");
}

// A second inclusion under a different stem must coexist (distinct modules).
mod nested {
    dsrs_macros::include_program!("tests/programs/qa.dsrs");

    #[test]
    fn nested_module_scope_works() {
        assert_eq!(&*qa::program().meta.name, "qa");
    }
}
