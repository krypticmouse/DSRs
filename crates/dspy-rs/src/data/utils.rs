use regex::Regex;
use std::sync::LazyLock;

#[allow(dead_code)]
static IS_URL_PAT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        "((http|https)://)(www.)?[a-zA-Z0-9@:%._\\+~#?&//=]{2,256}\\.[a-z]{2,6}\\b([-a-zA-Z0-9@:%._\\+~#?&//=]*)",
    )
    .unwrap()
});

/// Returns `true` if the string looks like an HTTP(S) URL.
pub fn is_url(path: &str) -> bool {
    IS_URL_PAT.is_match(path)
}
