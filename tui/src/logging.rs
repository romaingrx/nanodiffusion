use std::path::Path;

use color_eyre::{Result, eyre::WrapErr};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::EnvFilter;

/// Install a file-backed `tracing` subscriber. The returned guard must be held
/// for the lifetime of the program — dropping it flushes and stops the writer.
/// Defaults to `info` unless `RUST_LOG` is set.
pub fn init(path: &Path) -> Result<WorkerGuard> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create log dir {}", parent.display()))?;
    }

    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open log file {}", path.display()))?;

    let (writer, guard) = tracing_appender::non_blocking(file);
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("nanodiffusion_tui=info"));

    tracing_subscriber::fmt()
        .with_writer(writer)
        .with_env_filter(filter)
        .with_ansi(false)
        .with_target(false)
        .init();

    Ok(guard)
}
