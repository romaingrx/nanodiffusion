mod client;
mod effects;
mod logging;
mod model;
mod msg;
mod protocol;
mod render;
mod runtime;
mod session;
mod state;
mod terminal;
mod ui;
mod update;
mod view;

use std::{num::NonZeroU64, path::PathBuf};

use clap::Parser;
use color_eyre::Result;

use crate::{model::Model, state::SampleOptions};

#[derive(Parser, Debug)]
#[command(name = "nanodiffusion-tui", about)]
struct Args {
    /// Base URL of the nanodiffusion serve API.
    #[arg(long, default_value = "http://localhost:8000")]
    url: String,

    /// Number of diffusion steps. Defaults to the server's configured value.
    #[arg(long)]
    steps: Option<NonZeroU64>,

    /// Sampling temperature.
    #[arg(long)]
    temperature: Option<f64>,

    /// Top-k sampling (0 disables).
    #[arg(long)]
    top_k: Option<u64>,

    /// Top-p / nucleus sampling.
    #[arg(long)]
    top_p: Option<f64>,

    /// Maximum response length in tokens.
    #[arg(long)]
    max_length: Option<NonZeroU64>,

    /// Fixed RNG seed. Omit for a fresh stochastic sample each turn.
    #[arg(long)]
    seed: Option<i64>,

    /// Write logs to this file. Set `RUST_LOG` for verbosity (default `info`).
    #[arg(long, default_value = "tui.log")]
    log_file: PathBuf,
}

impl Args {
    const fn sample_options(&self) -> SampleOptions {
        SampleOptions {
            steps: self.steps,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            max_length: self.max_length,
            seed: self.seed,
        }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    color_eyre::install()?;
    install_panic_hook();
    let args = Args::parse();
    let _log_guard = logging::init(&args.log_file)?;
    tracing::info!(
        url = %args.url,
        log_file = %args.log_file.display(),
        "nanodiffusion-tui starting"
    );

    let http_client = client::build_client()?;
    let health = match client::fetch_health(&http_client, &args.url).await {
        Ok(h) => {
            tracing::info!(
                train_step = h.train_step,
                max_seq_len = u64::from(h.max_seq_len),
                vocab_size = u64::from(h.vocab_size),
                "server health ok"
            );
            Some(h)
        }
        Err(e) => {
            tracing::warn!(error = %e, "health check failed — proceeding offline");
            None
        }
    };

    let model = Model::new(args.url.clone(), args.sample_options(), health);
    let mut term = terminal::enter()?;
    let outcome = runtime::run(model, &mut term, http_client).await;
    terminal::leave(&mut term)?;
    if let Err(e) = &outcome {
        tracing::error!(error = %e, "run exited with error");
    }
    outcome
}

/// Restore the terminal before color-eyre's default panic handler runs. Without
/// this, a panic inside the event loop leaves the shell in alternate-screen +
/// raw mode.
fn install_panic_hook() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        terminal::restore();
        prev(info);
    }));
}
