use criterion::black_box;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;

fn bench_now(c: &mut Criterion) {
    // The first call will take some time for calibration
    quanta::Instant::now();

    let mut group = c.benchmark_group("Instant::now()");
    group.bench_function("fastant", |b| {
        b.iter(fastant::Instant::now);
    });
    group.bench_function("quanta", |b| {
        b.iter(quanta::Instant::now);
    });
    group.bench_function("std", |b| {
        b.iter(std::time::Instant::now);
    });
    group.finish();
}

fn bench_anchor_new(c: &mut Criterion) {
    c.bench_function("fastant::Anchor::new()", |b| {
        b.iter(fastant::Anchor::new);
    });
}

fn bench_as_unix_nanos(c: &mut Criterion) {
    let anchor = fastant::Anchor::new();
    c.bench_function("fastant::Instant::as_unix_nanos()", |b| {
        b.iter(|| {
            black_box(fastant::Instant::now().as_unix_nanos(&anchor));
        });
    });
}

criterion_group!(benches, bench_now, bench_anchor_new, bench_as_unix_nanos);
criterion_main!(benches);
