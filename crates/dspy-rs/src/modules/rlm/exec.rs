use pyo3::ffi::c_str;
use pyo3::types::{PyAnyMethods, PyDict, PyModule};
use pyo3::{Py, PyResult, Python};

use super::submit::{SUBMIT_STDOUT_ATTR, is_submit_terminated};

const NO_OUTPUT_MESSAGE: &str = "(no output - did you forget to print?)";

static EXEC_HELPER_CODE: &std::ffi::CStr = c_str!(
    r#"
import ast
import contextlib
import io


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
    let suppress_output = code.trim_end().ends_with(';');

    match run_exec(py, globals, code, suppress_output, max_output_chars) {
        Ok(output) => Ok(output),
        Err(err) => {
            let stdout = extract_submit_stdout(py, &err).unwrap_or_default();
            let traceback = format_python_traceback(py, &err).unwrap_or_else(|_| err.to_string());
            let combined = combine_stdout_and_traceback(stdout, traceback);
            Err(truncate_capture_output(&combined, max_output_chars))
        }
    }
}

fn run_exec(
    py: Python<'_>,
    globals: &Py<PyDict>,
    code: &str,
    suppress_output: bool,
    max_output_chars: usize,
) -> PyResult<String> {
    let module = PyModule::from_code(
        py,
        EXEC_HELPER_CODE,
        c_str!("<dsrs_exec>"),
        c_str!("dsrs_exec"),
    )?;
    let exec_fn = module.getattr("dsrs_exec")?;
    let globals = globals.bind(py);
    match exec_fn.call1((code, globals, suppress_output)) {
        Ok(result) => {
            let (stdout, repr): (String, Option<String>) = result.extract()?;
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

    format!("{head}\n... (truncated)\n{tail}")
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
            assert!(output.contains("... (truncated)"));
            assert!(output.starts_with("abcde"));
            assert!(output.ends_with("wxyz\n"));
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

            assert!(err.contains("... (truncated)"));
            assert!(err.chars().count() > 20);
        });
    }
}
