mod app;
mod terminal;

use anyhow::Result;
use clap::Parser;

use crate::app::App;

#[derive(Parser, Debug)]
#[command(name = "nanodiffusion-tui", about)]
struct Args {
    #[arg(long, default_value = "http://localhost:8000")]
    url: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install().map_err(|e| anyhow::anyhow!(Box::new(e)))?;
    let args = Args::parse();
    let mut term = terminal::enter()?;
    let outcome = App::new(args.url).run(&mut term).await;
    terminal::leave(&mut term)?;
    outcome
}
