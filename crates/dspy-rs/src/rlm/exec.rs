#![cfg(feature = "rlm")]

use pyo3::ffi::c_str;
use pyo3::types::{PyAnyMethods, PyDict, PyModule};
use pyo3::{Py, PyResult, Python};

const NO_OUTPUT_MESSAGE: &str = "(no output - did you forget to print?)";

static EXEC_HELPER_CODE: &std::ffi::CStr = c_str!(r#"
import ast
import contextlib
import io


def dsrs_exec(code, globals_dict, suppress_output):
    buffer = io.StringIO()
    result = None
    with contextlib.redirect_stdout(buffer):
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
    return buffer.getvalue(), (None if result is None else repr(result))
"#);

pub fn execute_repl_code(
    globals: &Py<PyDict>,
    code: &str,
    max_output_chars: usize,
) -> Result<String, String> {
    let suppress_output = code.trim_end().ends_with(';');

    Python::attach(|py| -> PyResult<String> {
        let module = PyModule::from_code(
            py,
            EXEC_HELPER_CODE,
            c_str!("<dsrs_exec>"),
            c_str!("dsrs_exec"),
        )?;
        let exec_fn = module.getattr("dsrs_exec")?;
        let globals = globals.bind(py);
        let (stdout, repr): (String, Option<String>) = exec_fn
            .call1((code, globals, suppress_output))?
            .extract()?;

        Ok(format_output(stdout, repr, max_output_chars))
    })
    .map_err(|err| err.to_string())
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

    let truncated: String = text.chars().take(max_chars).collect();
    format!("{truncated}\n... (truncated)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executes_expression_and_returns_repr() {
        let output = Python::attach(|py| {
            let globals = PyDict::new(py).unbind();
            execute_repl_code(&globals, "1 + 2", 100)
        })
        .expect("exec");

        assert_eq!(output, "3");
    }

    #[test]
    fn combines_stdout_and_repr() {
        let output = Python::attach(|py| {
            let globals = PyDict::new(py).unbind();
            execute_repl_code(&globals, "print('hi')\n2 + 3", 100)
        })
        .expect("exec");

        assert_eq!(output, "hi\n5");
    }

    #[test]
    fn suppresses_output_on_trailing_semicolon() {
        let output = Python::attach(|py| {
            let globals = PyDict::new(py).unbind();
            execute_repl_code(&globals, "2 + 3;", 100)
        })
        .expect("exec");

        assert_eq!(output, NO_OUTPUT_MESSAGE);
    }

    #[test]
    fn returns_error_for_invalid_code() {
        let result = Python::attach(|py| {
            let globals = PyDict::new(py).unbind();
            execute_repl_code(&globals, "def", 100)
        });

        assert!(result.is_err());
    }

    #[test]
    fn truncates_with_dspy_marker() {
        let output = Python::attach(|py| {
            let globals = PyDict::new(py).unbind();
            execute_repl_code(&globals, "print('abcdef')", 3)
        })
        .expect("exec");

        assert_eq!(output, "abc\n... (truncated)");
    }
}
