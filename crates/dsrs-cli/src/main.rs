//! The `dsrs` binary: argument parsing + exit codes over the library
//! functions in [`dsrs_cli`]. Keep logic out of here — tests exercise the
//! library directly.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use dsrs_cli::fmt::FmtOutcome;
use dsrs_cli::serve::ServeConfig;

#[derive(Parser)]
#[command(
    name = "dsrs",
    version,
    about = "DSRs .dsrs program toolchain: check, fmt, serve (RFC 0002 IR-7)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Parse + validate a .dsrs artifact; errors carry line/column positions.
    Check {
        /// Path to the .dsrs artifact.
        program: PathBuf,
    },
    /// Print (or rewrite) the canonical form of a .dsrs artifact.
    Fmt {
        /// Path to the .dsrs artifact.
        program: PathBuf,
        /// Rewrite the file in place instead of printing to stdout.
        #[arg(long)]
        write: bool,
    },
    /// Serve a .dsrs program over HTTP (POST /run, GET /schema, GET /program,
    /// GET /healthz).
    Serve {
        /// Path to the .dsrs artifact.
        program: PathBuf,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Overlay JSON in the named form ({"<param path>": <value>, ...}).
        #[arg(long)]
        overlay: Option<PathBuf>,
        /// Capability grant (repeatable): --allow net:search --allow fs:read.
        #[arg(long = "allow", value_name = "CAP")]
        allow: Vec<String>,
    },
}

fn main() -> ExitCode {
    match Cli::parse().cmd {
        Cmd::Check { program } => match dsrs_cli::check::check_file(&program) {
            Ok(report) => {
                println!("{report}");
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("{err}");
                ExitCode::FAILURE
            }
        },
        Cmd::Fmt { program, write } => match dsrs_cli::fmt::fmt_file(&program, write) {
            Ok(FmtOutcome::Canonical(text)) => {
                print!("{text}");
                ExitCode::SUCCESS
            }
            Ok(FmtOutcome::Rewrote) => {
                eprintln!("formatted `{}`", program.display());
                ExitCode::SUCCESS
            }
            Ok(FmtOutcome::Unchanged) => {
                eprintln!("`{}` already canonical", program.display());
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("{err:#}");
                ExitCode::FAILURE
            }
        },
        Cmd::Serve {
            program,
            host,
            port,
            overlay,
            allow,
        } => {
            let config = ServeConfig {
                program,
                overlay,
                allow,
            };
            let runtime = match tokio::runtime::Runtime::new() {
                Ok(runtime) => runtime,
                Err(err) => {
                    eprintln!("failed to start async runtime: {err}");
                    return ExitCode::FAILURE;
                }
            };
            match runtime.block_on(dsrs_cli::serve::serve(&config, &host, port)) {
                Ok(()) => ExitCode::SUCCESS,
                Err(err) => {
                    eprintln!("{err:#}");
                    ExitCode::FAILURE
                }
            }
        }
    }
}
