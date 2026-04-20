mod app;
mod client;
mod effects;
mod protocol;
mod render;
mod session;
mod state;
mod terminal;
mod ui;

use std::num::NonZeroU64;

use anyhow::{Context, Result};
use clap::Parser;

use crate::{app::App, state::SampleOptions};

#[derive(Parser, Debug)]
#[command(name = "nanodiffusion-tui", about)]
struct Args {
    /// Base URL of the nanodiffusion serve API.
    #[arg(long, default_value = "http://localhost:8000")]
    url: String,

    /// Number of diffusion steps (must be > 0). Defaults to the server's sample_defaults.
    #[arg(long)]
    steps: Option<u64>,

    /// Sampling temperature.
    #[arg(long)]
    temperature: Option<f64>,

    /// Top-k sampling (0 disables).
    #[arg(long)]
    top_k: Option<u64>,

    /// Top-p / nucleus sampling.
    #[arg(long)]
    top_p: Option<f64>,

    /// Maximum response length in tokens (must be > 0).
    #[arg(long)]
    max_length: Option<u64>,

    /// Fixed RNG seed. Omit for a fresh stochastic sample each turn.
    #[arg(long)]
    seed: Option<i64>,
}

impl Args {
    fn into_options(self) -> Result<(String, SampleOptions)> {
        let opts = SampleOptions {
            steps: positive("--steps", self.steps)?,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            max_length: positive("--max-length", self.max_length)?,
            seed: self.seed,
        };
        Ok((self.url, opts))
    }
}

fn positive(label: &str, value: Option<u64>) -> Result<Option<NonZeroU64>> {
    value
        .map(|n| NonZeroU64::new(n).with_context(|| format!("{label} must be > 0")))
        .transpose()
}

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install().map_err(|e| anyhow::anyhow!(Box::new(e)))?;
    let (url, opts) = Args::parse().into_options()?;
    let mut term = terminal::enter()?;
    let outcome = App::new(url, opts).run(&mut term).await;
    terminal::leave(&mut term)?;
    outcome
}
