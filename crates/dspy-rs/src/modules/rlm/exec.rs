use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi::c_str;
use pyo3::types::{PyAnyMethods, PyDict, PyModule};
use pyo3::{Py, PyResult, Python};

use super::submit::{SUBMIT_STDOUT_ATTR, is_submit_terminated};

const NO_OUTPUT_MESSAGE: &str = "(no output - did you forget to print?)";
const TRACEBACK_ATTR: &str = "__dsrs_traceback__";

static EXEC_HELPER_CODE: &std::ffi::CStr = c_str!(
    r#"
import ast
import contextlib
import io
import traceback


def dsrs_exec(code, globals_dict, suppress_output):
    buffer = io.StringIO()
    result = None
    with contextlib.redirect_stdout(buffer):
        try:
            parsed = ast.parse(code, mode="exec")
            if suppress_output or not parsed.body:
                exec(compile(parsed, "<repl>", "exec"), globals_dict, globals_dict)
            else:
                last = parsed.body[-1]
                if isinstance(last, ast.Expr):
                    body = parsed.body[:-1]
                    if body:
                        exec(
                            compile(ast.Module(body=body, type_ignores=[]), "<repl>", "exec"),
                            globals_dict,
                            globals_dict,
                        )
                    result = eval(
                        compile(ast.Expression(last.value), "<repl>", "eval"),
                        globals_dict,
                        globals_dict,
                    )
                else:
                    exec(compile(parsed, "<repl>", "exec"), globals_dict, globals_dict)
        except BaseException as exc:
            try:
                setattr(exc, "__dsrs_stdout__", buffer.getvalue())
            except Exception:
                pass
            try:
                setattr(exc, "__dsrs_traceback__", traceback.format_exc())
            except Exception:
                pass
            raise
    return buffer.getvalue(), (None if result is None else repr(result))
"#
);

pub fn execute_repl_code(
    py: Python<'_>,
    globals: &Py<PyDict>,
    code: &str,
    max_output_chars: usize,
) -> Result<String, String> {
    let prepared_code = preprocess_repl_code(code);
    let suppress_output = prepared_code.trim_end().ends_with(';');

    match run_exec(
        py,
        globals,
        &prepared_code,
        suppress_output,
        max_output_chars,
    ) {
        Ok(output) => Ok(output),
        Err(err) => {
            if let Some(repaired_code) = maybe_repair_submit_code(py, &prepared_code, &err) {
                match run_exec(
                    py,
                    globals,
                    &repaired_code,
                    suppress_output,
                    max_output_chars,
                ) {
                    Ok(output) => return Ok(output),
                    Err(_repaired_err) => {}
                }
            }
            let stdout = extract_submit_stdout(py, &err).unwrap_or_default();
            let traceback = extract_traceback(py, &err)
                .or_else(|| format_python_traceback(py, &err).ok())
                .unwrap_or_else(|| err.to_string());
            let combined = combine_stdout_and_traceback(stdout, traceback);
            Err(truncate_capture_output(&combined, max_output_chars))
        }
    }
}

fn maybe_repair_submit_code(py: Python<'_>, code: &str, err: &pyo3::PyErr) -> Option<String> {
    if !code.contains("SUBMIT(") {
        return None;
    }

    let traceback = extract_traceback(py, err).or_else(|| format_python_traceback(py, err).ok())?;
    if !traceback.contains("SyntaxError")
        || (!traceback.contains("unterminated triple-quoted string literal")
            && !traceback.contains("unterminated string literal"))
    {
        return None;
    }

    repair_submit_code(code)
}

fn repair_submit_code(code: &str) -> Option<String> {
    if !code.contains("SUBMIT(") {
        return None;
    }

    let mut repaired = code.trim_end().to_string();
    let mut changed = false;

    for quote in ["\"\"\"", "'''"] {
        if repaired.matches(quote).count() % 2 != 0 {
            repaired.push_str(quote);
            changed = true;
        }
    }

    if let Some(submit_start) = repaired.rfind("SUBMIT(") {
        let tail = &repaired[submit_start..];
        let open_parens = tail.chars().filter(|&c| c == '(').count();
        let close_parens = tail.chars().filter(|&c| c == ')').count();
        if open_parens > close_parens {
            repaired.push_str(&")".repeat(open_parens - close_parens));
            changed = true;
        }
    }

    if changed { Some(repaired) } else { None }
}

fn preprocess_repl_code(code: &str) -> String {
    let without_fences = strip_markdown_fence_lines(code);
    strip_leading_non_python_lines(&without_fences)
}

fn strip_markdown_fence_lines(text: &str) -> String {
    text.lines()
        .filter(|line| !line.trim_start().starts_with("```"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn strip_leading_non_python_lines(text: &str) -> String {
    let lines = text.lines().collect::<Vec<_>>();
    let first_code_index = lines.iter().position(|line| {
        let trimmed = line.trim();
        !trimmed.is_empty() && looks_like_python_line(trimmed)
    });

    let selected = match first_code_index {
        Some(index) => lines[index..].join("\n"),
        None => text.to_string(),
    };
    selected.trim_end().to_string()
}

fn looks_like_python_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed.starts_with('#')
        || trimmed.starts_with('[')
        || trimmed.starts_with('{')
        || trimmed.contains('=')
        || trimmed.contains('(')
    {
        return true;
    }

    let lower = trimmed.to_ascii_lowercase();
    for prefix in [
        "import ", "from ", "print", "def ", "for ", "if ", "while ", "try:", "with ", "class ",
    ] {
        if lower.starts_with(prefix) {
            return true;
        }
    }

    false
}

fn run_exec(
    py: Python<'_>,
    globals: &Py<PyDict>,
    code: &str,
    suppress_output: bool,
    max_output_chars: usize,
) -> PyResult<String> {
    let helper_globals = PyDict::new(py);
    py.run(
        EXEC_HELPER_CODE,
        Some(&helper_globals),
        Some(&helper_globals),
    )?;
    let exec_fn = helper_globals
        .get_item("dsrs_exec")
        .map_err(|_| PyRuntimeError::new_err("dsrs_exec helper function missing"))?;
    let globals = globals.bind(py);
    match exec_fn.call1((code, globals, suppress_output)) {
        Ok(result) => {
            let (stdout, repr) = result.extract::<(String, Option<String>)>()?;
            Ok(format_output(stdout, repr, max_output_chars))
        }
        Err(err) if is_submit_terminated(&err, py) => {
            let stdout = extract_submit_stdout(py, &err).unwrap_or_default();
            Ok(format_output(stdout, None, max_output_chars))
        }
        Err(err) => Err(err),
    }
}

fn extract_submit_stdout(py: Python<'_>, err: &pyo3::PyErr) -> Option<String> {
    err.value(py)
        .getattr(SUBMIT_STDOUT_ATTR)
        .ok()
        .and_then(|value| value.extract::<String>().ok())
}

fn extract_traceback(py: Python<'_>, err: &pyo3::PyErr) -> Option<String> {
    err.value(py)
        .getattr(TRACEBACK_ATTR)
        .ok()
        .and_then(|value| value.extract::<String>().ok())
}

fn format_python_traceback(py: Python<'_>, err: &pyo3::PyErr) -> PyResult<String> {
    let traceback = PyModule::import(py, "traceback")?;
    let formatted = traceback.getattr("format_exception")?.call1((
        err.get_type(py),
        err.value(py),
        err.traceback(py),
    ))?;
    let parts: Vec<String> = formatted.extract()?;
    Ok(parts.join(""))
}

fn combine_stdout_and_traceback(stdout: String, traceback: String) -> String {
    if stdout.is_empty() {
        return traceback;
    }
    if stdout.ends_with('\n') {
        format!("{stdout}{traceback}")
    } else {
        format!("{stdout}\n{traceback}")
    }
}

fn format_output(stdout: String, repr: Option<String>, max_chars: usize) -> String {
    let mut output = stdout;
    if let Some(repr) = repr {
        if !output.is_empty() && !output.ends_with('\n') {
            output.push('\n');
        }
        output.push_str(&repr);
    }

    if output.is_empty() {
        output = NO_OUTPUT_MESSAGE.to_string();
    }

    truncate_capture_output(&output, max_chars)
}

fn truncate_capture_output(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let total = text.chars().count();
    if total <= max_chars {
        return text.to_string();
    }

    let head_len = max_chars / 2;
    let tail_len = max_chars.saturating_sub(head_len);

    let head: String = text.chars().take(head_len).collect();
    let tail: String = text.chars().skip(total.saturating_sub(tail_len)).collect();
    let truncation_notice = format!("... [STDOUT TRUNCATED: Exceeded {max_chars} char threshold]");

    format!("{head}\n{tail}\n{truncation_notice}")
}

#[cfg(test)]
mod tests {
    use pyo3::types::{PyDict, PyDictMethods};

    use super::*;
    use crate::modules::rlm::submit::SubmitTerminated;

    #[test]
    fn executes_expression_and_returns_repr() {
        Python::attach(|py| {
            let globals = PyDict::new(py).unbind();
            let output = execute_repl_code(py, &globals, "1 + 2", 100).expect("exec");
            assert_eq!(output, "3");
        });
    }

    #[test]
    fn combines_stdout_and_repr() {
        Python::attach(|py| {
            let globals = PyDict::new(py).unbind();
            let output = execute_repl_code(py, &globals, "print('hi')\n2 + 3", 100).expect("exec");
            assert_eq!(output, "hi\n5");
        });
    }

    #[test]
    fn suppresses_output_on_trailing_semicolon() {
        Python::attach(|py| {
            let globals = PyDict::new(py).unbind();
            let output = execute_repl_code(py, &globals, "2 + 3;", 100).expect("exec");
            assert_eq!(output, NO_OUTPUT_MESSAGE);
        });
    }

    #[test]
    fn returns_no_output_message_when_no_stdout_or_repr() {
        Python::attach(|py| {
            let globals = PyDict::new(py).unbind();
            let output = execute_repl_code(py, &globals, "x = 10", 100).expect("exec");
            assert_eq!(output, NO_OUTPUT_MESSAGE);
        });
    }

    #[test]
    fn truncates_with_head_and_tail() {
        Python::attach(|py| {
            let globals = PyDict::new(py).unbind();
            let output = execute_repl_code(py, &globals, "print('abcdefghijklmnopqrstuvwxyz')", 10)
                .expect("exec");
            assert!(output.contains("... [STDOUT TRUNCATED: Exceeded 10 char threshold]"));
            assert!(output.starts_with("abcde"));
            assert!(output.contains("wxyz\n"));
            assert!(output.ends_with("... [STDOUT TRUNCATED: Exceeded 10 char threshold]"));
        });
    }

    #[test]
    fn submit_terminated_is_treated_as_success_path() {
        Python::attach(|py| {
            let globals = PyDict::new(py);
            globals
                .set_item("SubmitTerminated", py.get_type::<SubmitTerminated>())
                .expect("set type");
            let globals = globals.unbind();

            let output = execute_repl_code(
                py,
                &globals,
                "print('before submit')\nraise SubmitTerminated('done')",
                200,
            )
            .expect("exec");

            assert_eq!(output, "before submit\n");
        });
    }

    #[test]
    fn syntax_errors_return_err_string() {
        Python::attach(|py| {
            let globals = PyDict::new(py).unbind();
            let err = execute_repl_code(py, &globals, "if True print('x')", 100)
                .expect_err("should fail");
            assert!(err.contains("SyntaxError"));
            assert!(err.contains("Traceback"));
        });
    }

    #[test]
    fn includes_stdout_and_traceback_on_python_errors() {
        Python::attach(|py| {
            let globals = PyDict::new(py).unbind();
            let err = execute_repl_code(
                py,
                &globals,
                "print('before failure')\nraise ValueError('boom')",
                500,
            )
            .expect_err("should fail");

            assert!(err.contains("before failure"));
            assert!(err.contains("Traceback"));
            assert!(err.contains("ValueError: boom"));
        });
    }

    #[test]
    fn import_errors_include_traceback_and_exception_type() {
        Python::attach(|py| {
            let globals = PyDict::new(py).unbind();
            let err = execute_repl_code(py, &globals, "import definitely_missing_module_xyz", 500)
                .expect_err("should fail");

            assert!(err.contains("Traceback"));
            assert!(
                err.contains("ModuleNotFoundError")
                    || err.contains("ImportError")
                    || err.contains("AttributeError"),
                "expected import-related failure class in traceback: {err}"
            );
            assert!(
                err.contains("definitely_missing_module_xyz")
                    || err.contains("partially initialized module"),
                "expected import target or fallback import error context: {err}"
            );
        });
    }

    #[test]
    fn truncates_error_output_with_budget() {
        Python::attach(|py| {
            let globals = PyDict::new(py).unbind();
            let err = execute_repl_code(
                py,
                &globals,
                "print('abcdefghijklmnopqrstuvwxyz')\nraise RuntimeError('abcdefghijklmnopqrstuvwxyz')",
                20,
            )
            .expect_err("should fail");

            assert!(err.contains("... [STDOUT TRUNCATED: Exceeded 20 char threshold]"));
            assert!(err.chars().count() > 20);
        });
    }

    #[test]
    fn truncation_is_unicode_safe_for_multibyte_characters() {
        let text = "😀".repeat(40);
        let truncated = truncate_capture_output(&text, 9);

        assert!(truncated.contains("... [STDOUT TRUNCATED: Exceeded 9 char threshold]"));
        assert!(truncated.is_char_boundary(truncated.len()));
        assert!(truncated.contains('😀'));
    }

    #[test]
    fn preprocess_strips_markdown_fences() {
        let raw = "```python\nprint('a')\n```\n```py\nprint('b')\n```\n```\nprint('c')\n```";
        let prepared = preprocess_repl_code(raw);
        assert_eq!(prepared, "print('a')\nprint('b')\nprint('c')");
    }

    #[test]
    fn preprocess_strips_leading_prose_until_python() {
        let raw = "Let me start by exploring the data first.\n\n```python\n# First, inspect\na = 1\nprint(a)\n```";
        let prepared = preprocess_repl_code(raw);
        assert_eq!(prepared, "# First, inspect\na = 1\nprint(a)");
    }

    #[test]
    fn execute_repl_code_handles_failed_turn_one_pattern() {
        Python::attach(|py| {
            let globals = PyDict::new(py).unbind();
            let raw = "Let me start by exploring the data to understand the structure and then systematically find recurring corrections.\n\n```python\n# First, explore the data structure\nprint('ok')\n```";
            let output = execute_repl_code(py, &globals, raw, 500).expect("exec");
            assert_eq!(output, "ok\n");
        });
    }

    #[test]
    fn repair_submit_code_closes_unterminated_triple_quote_and_paren() {
        let repaired = repair_submit_code("SUBMIT(direct_answer=\"\"\"hello")
            .expect("repair should produce code");
        assert_eq!(repaired, "SUBMIT(direct_answer=\"\"\"hello\"\"\")");
    }

    #[test]
    fn execute_repl_code_repairs_unterminated_submit_payload() {
        Python::attach(|py| {
            let globals = PyDict::new(py);
            globals
                .set_item("SubmitTerminated", py.get_type::<SubmitTerminated>())
                .expect("set type");
            py.run(
                c_str!("def SUBMIT(**kwargs):\n    raise SubmitTerminated('done')\n"),
                Some(&globals),
                Some(&globals),
            )
            .expect("submit helper");
            let globals = globals.unbind();

            let output = execute_repl_code(py, &globals, "SUBMIT(direct_answer=\"\"\"hello", 500)
                .expect("submit should recover");
            assert_eq!(output, NO_OUTPUT_MESSAGE);
        });
    }
}
