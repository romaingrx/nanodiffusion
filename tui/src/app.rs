use std::time::Duration;

use anyhow::Result;
use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyModifiers};
use futures::StreamExt;
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph, Wrap},
};
use tokio::{sync::mpsc, task::JoinHandle, time::interval};

use crate::{
    client::{ClientEvent, stream_chat},
    protocol::{ChatRequest, Message, Role, StreamFrame},
    terminal::Tui,
};

pub struct App {
    base_url: String,
    history: Vec<Message>,
    input: String,
    stream: Option<Stream>,
    status: String,
    should_quit: bool,
}

struct Stream {
    rx: mpsc::Receiver<ClientEvent>,
    handle: JoinHandle<Result<()>>,
    latest: Option<StreamFrame>,
}

impl App {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url,
            history: Vec::new(),
            input: String::new(),
            stream: None,
            status: "ready — type a message, enter to send, esc to cancel, ctrl-c to quit".into(),
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
                evt = events.next() => match evt {
                    Some(Ok(e)) => self.on_event(e),
                    Some(Err(e)) => return Err(e.into()),
                    None => self.should_quit = true,
                },
                msg = recv_stream(&mut self.stream) => self.on_stream(msg),
            }
        }
        if let Some(s) = self.stream.take() {
            s.handle.abort();
        }
        Ok(())
    }

    fn on_event(&mut self, evt: Event) {
        let Event::Key(KeyEvent { code, modifiers, .. }) = evt else { return };
        if modifiers.contains(KeyModifiers::CONTROL) && matches!(code, KeyCode::Char('c')) {
            self.should_quit = true;
            return;
        }
        match code {
            KeyCode::Esc => {
                if self.stream.is_some() {
                    self.cancel();
                } else {
                    self.should_quit = true;
                }
            }
            KeyCode::Enter if self.stream.is_none() && !self.input.trim().is_empty() => {
                self.send();
            }
            KeyCode::Backspace => {
                self.input.pop();
            }
            KeyCode::Char(c) if self.stream.is_none() => {
                self.input.push(c);
            }
            _ => {}
        }
    }

    fn send(&mut self) {
        let content = std::mem::take(&mut self.input);
        self.history.push(Message { role: Role::User, content });
        let req = ChatRequest {
            messages: self.history.clone(),
            max_length: None,
            seed: None,
            steps: None,
            temperature: None,
            top_k: None,
            top_p: None,
        };
        let (tx, rx) = mpsc::channel(64);
        let base_url = self.base_url.clone();
        let handle = tokio::spawn(async move { stream_chat(&base_url, &req, tx).await });
        self.status = "streaming…".into();
        self.stream = Some(Stream { rx, handle, latest: None });
    }

    fn cancel(&mut self) {
        if let Some(s) = self.stream.take() {
            s.handle.abort();
        }
        self.status = "cancelled".into();
    }

    fn on_stream(&mut self, msg: Option<ClientEvent>) {
        let Some(s) = self.stream.as_mut() else { return };
        match msg {
            Some(ClientEvent::Frame(frame)) => s.latest = Some(frame),
            Some(ClientEvent::Done) | None => {
                if let Some(frame) = s.latest.take() {
                    self.history.push(Message { role: Role::Assistant, content: frame.text });
                }
                self.stream = None;
                self.status = "ready".into();
            }
            Some(ClientEvent::Error(e)) => {
                self.stream = None;
                self.status = format!("error: {e}");
            }
        }
    }

    fn draw(&self, f: &mut Frame<'_>) {
        let [chat_area, input_area, status_area] = Layout::vertical([
            Constraint::Min(3),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .areas(f.area());

        self.draw_chat(f, chat_area);
        self.draw_input(f, input_area);
        self.draw_status(f, status_area);
    }

    fn draw_chat(&self, f: &mut Frame<'_>, area: Rect) {
        let mut lines: Vec<Line> = Vec::new();
        for msg in &self.history {
            lines.push(role_label(msg.role));
            for line in msg.content.lines() {
                lines.push(Line::from(line.to_string()));
            }
            lines.push(Line::from(""));
        }
        if let Some(s) = self.stream.as_ref() {
            lines.push(role_label(Role::Assistant));
            if let Some(frame) = s.latest.as_ref() {
                for line in frame.text.lines() {
                    lines.push(Line::from(line.to_string()));
                }
            } else {
                lines.push(Line::from(Span::styled("…", Style::default().fg(Color::DarkGray))));
            }
        }
        let block = Block::bordered().title(" chat ");
        let p = Paragraph::new(lines).wrap(Wrap { trim: false }).block(block);
        f.render_widget(p, area);
    }

    fn draw_input(&self, f: &mut Frame<'_>, area: Rect) {
        let block = Block::bordered().title(" message ");
        let style = if self.stream.is_some() {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default()
        };
        let p = Paragraph::new(self.input.as_str()).style(style).block(block);
        f.render_widget(p, area);
    }

    fn draw_status(&self, f: &mut Frame<'_>, area: Rect) {
        let stats = self.stream.as_ref().and_then(|s| s.latest.as_ref()).map(|frame| {
            let masks = frame.mask_positions.len();
            let total = u64::from(frame.total);
            format!(" step {}/{} — {masks} masked ", frame.step, total)
        });
        let line = Line::from(vec![
            Span::styled(format!(" {} ", self.status), Style::default().add_modifier(Modifier::DIM)),
            Span::raw(stats.unwrap_or_default()),
        ]);
        f.render_widget(Paragraph::new(line), area);
    }
}

fn role_label(role: Role) -> Line<'static> {
    let (label, color) = match role {
        Role::User => ("you", Color::Cyan),
        Role::Assistant => ("model", Color::Magenta),
        Role::System => ("system", Color::Yellow),
    };
    Line::from(Span::styled(
        format!("{label}:"),
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    ))
}

async fn recv_stream(stream: &mut Option<Stream>) -> Option<ClientEvent> {
    match stream {
        Some(s) => s.rx.recv().await,
        None => std::future::pending().await,
    }
}
