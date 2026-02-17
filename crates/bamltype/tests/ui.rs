#[test]
#[cfg_attr(
    miri,
    ignore = "trybuild launches subprocesses and is unsupported under miri"
)]
fn ui_compile_failures() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
