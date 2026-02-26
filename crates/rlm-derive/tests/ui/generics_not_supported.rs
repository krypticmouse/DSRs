use rlm_derive::rlm_type;

#[rlm_type]
struct Generic<T> {
    value: T,
}

fn main() {}
