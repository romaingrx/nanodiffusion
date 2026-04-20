use std::time::Duration;

use anyhow::Result;
use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyModifiers};
use futures::StreamExt;
use ratatui::{
    Frame,
    layout::{Constraint, Layout},
};
use tokio::time::interval;

use crate::{
    effects::Reveal,
    render,
    session::{Session, SessionUpdate},
    state::{ChatState, SampleOptions},
    terminal::Tui,
    ui::{ChatPane, InputPane, StatusBar},
};

const TICK: Duration = Duration::from_millis(50);

pub struct App {
    base_url: String,
    sample_opts: SampleOptions,
    chat: ChatState,
    session: Option<Session>,
    reveal: Reveal,
    status: String,
    should_quit: bool,
}

impl App {
    pub fn new(base_url: String, sample_opts: SampleOptions) -> Self {
        Self {
            base_url,
            sample_opts,
            chat: ChatState::default(),
            session: None,
            reveal: Reveal::new(),
            status: "ready — type, enter to send, esc to cancel, ctrl-c to quit".into(),
            should_quit: false,
        }
    }

    pub async fn run(mut self, term: &mut Tui) -> Result<()> {
        let mut events = EventStream::new();
        let mut ticker = interval(TICK);
        while !self.should_quit {
            term.draw(|f| self.draw(f))?;
            tokio::select! {
                _ = ticker.tick() => {}
                maybe_evt = events.next() => self.on_raw_event(maybe_evt)?,
                update = poll_session(&mut self.session) => self.on_session_update(update),
            }
        }
        self.session = None;
        self.reveal.reset();
        Ok(())
    }

    fn on_raw_event(&mut self, evt: Option<std::io::Result<Event>>) -> Result<()> {
        match evt {
            Some(Ok(Event::Key(key))) => self.on_key(key),
            Some(Ok(_)) => {}
            Some(Err(e)) => return Err(e.into()),
            None => self.should_quit = true,
        }
        Ok(())
    }

    fn on_key(&mut self, key: KeyEvent) {
        match KeyAction::from(key, self.session.is_some(), self.chat.input_is_empty()) {
            KeyAction::Nothing => {}
            KeyAction::Quit => self.should_quit = true,
            KeyAction::Cancel => self.cancel(),
            KeyAction::Send => self.send(),
            KeyAction::Type(c) => self.chat.push_char(c),
            KeyAction::Backspace => self.chat.pop_char(),
        }
    }

    fn send(&mut self) {
        let req = self.chat.commit_user_turn(&self.sample_opts);
        self.session = Some(Session::spawn(self.base_url.clone(), req));
        self.reveal.reset();
        self.status = "streaming…".into();
    }

    fn cancel(&mut self) {
        self.session = None;
        self.reveal.reset();
        self.status = "cancelled".into();
    }

    fn on_session_update(&mut self, update: Option<SessionUpdate>) {
        match update {
            Some(SessionUpdate::Frame) => {
                if let Some(frame) = self.session.as_ref().and_then(Session::latest) {
                    self.reveal.observe(frame);
                }
            }
            Some(SessionUpdate::Done) | None => self.finalize(),
            Some(SessionUpdate::Error(e)) => self.fail(e),
        }
    }

    fn finalize(&mut self) {
        let body = self
            .session
            .take()
            .as_mut()
            .and_then(Session::take_latest)
            .map(|frame| render::finalized_body(&frame.text))
            .filter(|b| !b.is_empty());
        match body {
            Some(assistant) => self.chat.push_assistant(assistant),
            None => {
                // Nothing usable came back — drop the user message so we keep
                // a strict user/assistant/… alternation (the server rejects
                // anything else with 422) and give the user back their input.
                self.chat.rollback_last_user();
                self.status = "no response — input restored".into();
                self.reveal.reset();
                return;
            }
        }
        self.reveal.reset();
        self.status = "ready".into();
    }

    fn fail(&mut self, msg: String) {
        self.session = None;
        self.reveal.reset();
        self.chat.rollback_last_user();
        self.status = format!("error: {msg}");
    }

    fn draw(&mut self, f: &mut Frame<'_>) {
        let [chat_area, input_area, status_area] = Layout::vertical([
            Constraint::Min(3),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .areas(f.area());

        f.render_widget(
            ChatPane {
                history: self.chat.history(),
                streaming: self.session.as_ref().and_then(Session::latest),
            },
            chat_area,
        );
        f.render_widget(
            InputPane {
                buffer: self.chat.input(),
                locked: self.session.is_some(),
            },
            input_area,
        );
        f.render_widget(
            StatusBar {
                status: &self.status,
                progress: self.session.as_ref().and_then(Session::latest),
                tok_per_sec: self.session.as_ref().and_then(Session::tokens_per_second),
            },
            status_area,
        );

        self.reveal.apply(f.buffer_mut(), chat_area);
    }
}

enum KeyAction {
    Nothing,
    Quit,
    Cancel,
    Send,
    Type(char),
    Backspace,
}

impl KeyAction {
    fn from(key: KeyEvent, streaming: bool, input_empty: bool) -> Self {
        if key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('c')) {
            return Self::Quit;
        }
        match key.code {
            KeyCode::Esc if streaming => Self::Cancel,
            KeyCode::Esc => Self::Quit,
            KeyCode::Enter if !streaming && !input_empty => Self::Send,
            KeyCode::Backspace if !streaming => Self::Backspace,
            KeyCode::Char(c) if !streaming => Self::Type(c),
            _ => Self::Nothing,
        }
    }
}

async fn poll_session(session: &mut Option<Session>) -> Option<SessionUpdate> {
    match session {
        Some(s) => s.poll().await,
        None => std::future::pending().await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyEventKind;

    fn key(code: KeyCode, mods: KeyModifiers) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: mods,
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::NONE,
        }
    }

    #[test]
    fn ctrl_c_always_quits() {
        let a = KeyAction::from(key(KeyCode::Char('c'), KeyModifiers::CONTROL), true, false);
        assert!(matches!(a, KeyAction::Quit));
    }

    #[test]
    fn esc_cancels_while_streaming_else_quits() {
        assert!(matches!(
            KeyAction::from(key(KeyCode::Esc, KeyModifiers::NONE), true, false),
            KeyAction::Cancel
        ));
        assert!(matches!(
            KeyAction::from(key(KeyCode::Esc, KeyModifiers::NONE), false, false),
            KeyAction::Quit
        ));
    }

    #[test]
    fn enter_sends_only_when_idle_with_input() {
        assert!(matches!(
            KeyAction::from(key(KeyCode::Enter, KeyModifiers::NONE), false, false),
            KeyAction::Send
        ));
        assert!(matches!(
            KeyAction::from(key(KeyCode::Enter, KeyModifiers::NONE), true, false),
            KeyAction::Nothing
        ));
        assert!(matches!(
            KeyAction::from(key(KeyCode::Enter, KeyModifiers::NONE), false, true),
            KeyAction::Nothing
        ));
    }

    #[test]
    fn typing_is_blocked_during_stream() {
        assert!(matches!(
            KeyAction::from(key(KeyCode::Char('a'), KeyModifiers::NONE), false, true),
            KeyAction::Type('a')
        ));
        assert!(matches!(
            KeyAction::from(key(KeyCode::Char('a'), KeyModifiers::NONE), true, true),
            KeyAction::Nothing
        ));
    }
}
