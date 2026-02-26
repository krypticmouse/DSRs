use crate::TypeIR;

pub(crate) fn simplify_type_token(token: &str) -> String {
    token.rsplit("::").next().unwrap_or(token).to_string()
}

pub(crate) fn simplify_type_name_with(
    raw: &str,
    mut render_token: impl FnMut(&str) -> String,
) -> String {
    let mut result = String::with_capacity(raw.len());
    let mut chars = raw.chars();
    while let Some(ch) = chars.next() {
        if ch == '`' {
            let mut token = String::new();
            for next in chars.by_ref() {
                if next == '`' {
                    break;
                }
                token.push(next);
            }
            result.push_str(&render_token(&token));
        } else {
            result.push(ch);
        }
    }
    result
}

pub(crate) fn render_type_name_for_prompt_with(
    type_ir: &TypeIR,
    render_token: impl FnMut(&str) -> String,
) -> String {
    simplify_type_name_with(&type_ir.diagnostic_repr().to_string(), render_token)
        .replace("class ", "")
        .replace("enum ", "")
        .replace(" | ", " or ")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplify_type_name_with_rewrites_backtick_tokens() {
        let raw = "class `my::pkg::Thing` | `other::Foo`";
        let rendered = simplify_type_name_with(raw, simplify_type_token);
        assert_eq!(rendered, "class Thing | Foo");
    }
}
