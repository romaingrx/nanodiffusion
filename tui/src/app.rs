use std::time::Duration;

use anyhow::Result;
use crossterm::event::{Event, EventStream, KeyCode};
use futures::StreamExt;
use ratatui::{
    Frame,
    layout::Alignment,
    widgets::{Block, Paragraph},
};
use tokio::time::interval;

use crate::terminal::Tui;

pub struct App {
    url: String,
    should_quit: bool,
}

impl App {
    pub fn new(url: String) -> Self {
        Self {
            url,
            should_quit: false,
        }
    }

    pub async fn run(mut self, term: &mut Tui) -> Result<()> {
        let mut events = EventStream::new();
        let mut ticker = interval(Duration::from_millis(50));
        while !self.should_quit {
            term.draw(|f| self.draw(f))?;
            tokio::select! {
                _ = ticker.tick() => {}
                maybe_event = events.next() => {
                    match maybe_event {
                        Some(Ok(evt)) => self.on_event(evt),
                        Some(Err(e)) => return Err(e.into()),
                        None => self.should_quit = true,
                    }
                }
            }
        }
        Ok(())
    }

    fn draw(&self, f: &mut Frame<'_>) {
        let body = Paragraph::new(format!("connected to: {}\n\npress q or esc to quit", self.url))
            .alignment(Alignment::Center)
            .block(Block::bordered().title(" nanodiffusion-tui "));
        f.render_widget(body, f.area());
    }

    fn on_event(&mut self, evt: Event) {
        if let Event::Key(k) = evt
            && matches!(k.code, KeyCode::Char('q') | KeyCode::Esc)
        {
            self.should_quit = true;
        }
    }
}
