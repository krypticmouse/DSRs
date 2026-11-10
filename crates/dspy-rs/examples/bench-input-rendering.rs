//! Fixed-workload prompt rendering benchmark. No provider or dataset downloads.
//! Run in release mode with --no-default-features; --alloc profiles allocations
//! separately from wall-clock runs. --dump DIR retains byte-comparison outputs.
use dspy_rs::ir::{FieldDef, RenderSpec, SignatureDef};
use dspy_rs::{ChatAdapter, FieldType, JsonMap};
use serde_json::{Value, json};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

struct Allocator;
static COUNT: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let result = unsafe { System.alloc(layout) };
        if COUNT.load(Ordering::Relaxed) && !result.is_null() {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        result
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let result = unsafe { System.realloc(ptr, layout, size) };
        if COUNT.load(Ordering::Relaxed) && !result.is_null() {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(size as u64, Ordering::Relaxed);
        }
        result
    }
}

fn fixture(width: usize, jinja: bool) -> (SignatureDef, JsonMap) {
    let mut fields = vec![FieldDef::new("payload", FieldType::String)];
    let mut input = JsonMap::new();
    input.insert("payload".into(), Value::String("context ".repeat(4096)));
    for i in 0..width {
        let name = format!("f{i}");
        let mut field = FieldDef::new(&name, FieldType::String);
        if i == 0 {
            field = field.aliased("query");
        }
        if jinja {
            field.render = RenderSpec::Jinja("{{ field.name }}|{{ this }}|{{ input.query }}|{{ input.payload|length }}|{{ vars|length }}".into());
        }
        fields.push(field);
        input.insert(name, Value::String(format!("{i}:{}", "x".repeat(512))));
    }
    (
        SignatureDef {
            name: "RenderBench".into(),
            instruction: "Render the input.".into(),
            inputs: fields.into_boxed_slice(),
            outputs: vec![FieldDef::new("answer", FieldType::String)].into_boxed_slice(),
        },
        input,
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let iters: u64 = args
        .first()
        .filter(|s| !s.starts_with('-'))
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(2000);
    if iters == 0 {
        return Err("iterations must be positive".into());
    }
    let profile = args.iter().any(|s| s == "--alloc");
    let dump = args
        .iter()
        .position(|s| s == "--dump")
        .map(|i| args.get(i + 1).ok_or("--dump requires a directory"))
        .transpose()?;
    if let Some(path) = dump {
        std::fs::create_dir_all(path)?;
    }
    let adapter = ChatAdapter;
    let mut records = Vec::new();
    for jinja in [false, true] {
        for width in [1, 8, 16] {
            let (def, input) = fixture(width, jinja);
            let expected = adapter.format_input_def(&def, &input);
            for _ in 0..100 {
                std::hint::black_box(adapter.format_input_def(&def, &input));
            }
            ALLOCS.store(0, Ordering::Relaxed);
            BYTES.store(0, Ordering::Relaxed);
            COUNT.store(profile, Ordering::Relaxed);
            let started = Instant::now();
            for _ in 0..iters {
                std::hint::black_box(adapter.format_input_def(&def, &input));
            }
            let elapsed = started.elapsed();
            COUNT.store(false, Ordering::Relaxed);
            let allocs = ALLOCS.load(Ordering::Relaxed);
            let bytes = BYTES.load(Ordering::Relaxed);
            let label = format!("{}-{width}", if jinja { "jinja" } else { "default" });
            if let Some(path) = dump {
                std::fs::write(
                    std::path::Path::new(path).join(format!("{label}.txt")),
                    &expected,
                )?;
            }
            records.push(json!({"case":label,"iterations":iters,"warmup":100,
                "ns_per_op":elapsed.as_nanos() as f64 / iters as f64,
                "allocation_profile":profile,"allocations_per_op":allocs as f64 / iters as f64,
                "allocated_bytes_per_op":bytes as f64 / iters as f64,"output_bytes":expected.len(),
                "output_hash":format!("{:016x}",dspy_rs::utils::hash::stable_hash_debug(&expected))}));
        }
    }
    println!("{}", serde_json::to_string(&records)?);
    Ok(())
}
