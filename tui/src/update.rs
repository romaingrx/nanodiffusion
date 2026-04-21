//! Pure reducer — the `U` in MVU.
//!
//! Given the current `Model` and a `Msg`, it advances state and returns a
//! `Cmd` describing any side effect it wants. No I/O, no tokio, no `Instant::now`
//! except when a state transition genuinely begins *now* (see
//! [`on_frame`]'s `started: Instant::now()` — the canonical TEA rule is "no
//! clock reads except on transitions triggered by user input").
//!
//! Everything here is unit-testable: construct a `Model`, send a `Msg`, assert
//! on the returned `Cmd` and resulting `Model` fields.

use std::time::Instant;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::{
    model::{Model, Status, StreamingState},
    msg::{Cmd, Msg},
    protocol::StreamFrame,
    render,
};

/// Safety cushion on top of the server's `max_length` so tokens consumed by
/// special-token framing don't tip us over at commit time.
const PROMPT_SAFETY_BUFFER: u64 = 8;

pub fn update(m: &mut Model, msg: Msg) -> Cmd {
    match msg {
        Msg::Tick => {
            m.tick = m.tick.wrapping_add(1);
            Cmd::None
        }
        Msg::Key(key) => on_key(m, key),
        Msg::Frame(frame) => {
            on_frame(m, frame);
            Cmd::None
        }
        Msg::StreamDone => finalize(m),
        Msg::StreamFailed(err) => fail(m, err.to_string()),
        Msg::TerminalClosed => {
            m.should_quit = true;
            Cmd::Quit
        }
    }
}

fn on_key(m: &mut Model, key: KeyEvent) -> Cmd {
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        return match key.code {
            KeyCode::Char('c') => {
                m.should_quit = true;
                Cmd::Quit
            }
            KeyCode::Char('l') if !m.is_streaming() => {
                m.chat.clear_history();
                m.status = Status::notice("history cleared");
                Cmd::None
            }
            _ => Cmd::None,
        };
    }
    match key.code {
        KeyCode::Esc if m.is_streaming() => cancel(m),
        KeyCode::Esc => {
            m.should_quit = true;
            Cmd::Quit
        }
        KeyCode::Enter if !m.is_streaming() && !m.chat.input_is_empty() => send(m),
        KeyCode::Backspace if !m.is_streaming() => {
            m.chat.pop_char();
            Cmd::None
        }
        KeyCode::Char(c) if !m.is_streaming() => {
            m.chat.push_char(c);
            Cmd::None
        }
        _ => Cmd::None,
    }
}

fn send(m: &mut Model) -> Cmd {
    if let Some(max) = m.sample_opts.resolved_max_length(m.health.as_ref()) {
        let estimate = m.chat.estimate_prompt_tokens();
        if estimate >= max.saturating_sub(PROMPT_SAFETY_BUFFER) {
            m.status = Status::error(format!(
                "prompt ~{estimate} tok ≥ max_length {max} — ctrl-l to reset"
            ));
            return Cmd::None;
        }
    }
    m.chat.commit_user_turn();
    let req = m.sample_opts.build_request(m.chat.history().to_vec());
    m.status = Status::streaming();
    Cmd::Spawn {
        url: m.base_url.clone(),
        req,
    }
}

fn cancel(m: &mut Model) -> Cmd {
    m.streaming = None;
    m.status = Status::notice("cancelled");
    Cmd::AbortStream
}

fn on_frame(m: &mut Model, frame: StreamFrame) {
    match &mut m.streaming {
        Some(state) => state.latest = frame,
        None => {
            m.streaming = Some(StreamingState {
                latest: frame,
                started: Instant::now(),
            });
        }
    }
}

fn finalize(m: &mut Model) -> Cmd {
    let body = m
        .streaming
        .take()
        .map(|s| render::finalized_body(&s.latest.text))
        .filter(|b| !b.is_empty());
    if let Some(text) = body {
        m.chat.push_assistant(text);
        m.status = Status::idle("ready");
    } else {
        m.chat.rollback_last_user();
        m.status = Status::notice("no response — input restored");
    }
    Cmd::None
}

fn fail(m: &mut Model, msg: String) -> Cmd {
    m.streaming = None;
    m.chat.rollback_last_user();
    m.status = Status::error(msg);
    Cmd::AbortStream
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use crossterm::event::{KeyEventKind, KeyEventState};

    use super::*;
    use crate::{client::ClientError, model::StatusKind, state::SampleOptions};

    fn model() -> Model {
        Model::new("http://x".into(), SampleOptions::default(), None)
    }

    fn key(code: KeyCode, mods: KeyModifiers) -> Msg {
        Msg::Key(KeyEvent {
            code,
            modifiers: mods,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        })
    }

    fn frame(step: u64, masks: &[i64]) -> StreamFrame {
        StreamFrame {
            step,
            total: NonZeroU64::new(32).unwrap(),
            text: "<|assistant_start|>hello<|assistant_end|>".into(),
            tokens: vec![],
            mask_positions: masks.to_vec(),
        }
    }

    #[test]
    fn ctrl_c_always_quits() {
        let mut m = model();
        let cmd = update(&mut m, key(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(m.should_quit);
        assert!(matches!(cmd, Cmd::Quit));
    }

    #[test]
    fn esc_while_idle_quits() {
        let mut m = model();
        let cmd = update(&mut m, key(KeyCode::Esc, KeyModifiers::NONE));
        assert!(m.should_quit);
        assert!(matches!(cmd, Cmd::Quit));
    }

    #[test]
    fn esc_while_streaming_cancels() {
        let mut m = model();
        m.streaming = Some(StreamingState {
            latest: frame(0, &[0, 1]),
            started: Instant::now(),
        });
        m.status = Status::streaming();
        let cmd = update(&mut m, key(KeyCode::Esc, KeyModifiers::NONE));
        assert!(!m.should_quit);
        assert!(m.streaming.is_none());
        assert_eq!(m.status.kind, StatusKind::Notice);
        assert!(matches!(cmd, Cmd::AbortStream));
    }

    #[test]
    fn typing_populates_input_when_idle() {
        let mut m = model();
        let _ = update(&mut m, key(KeyCode::Char('h'), KeyModifiers::NONE));
        let _ = update(&mut m, key(KeyCode::Char('i'), KeyModifiers::NONE));
        assert_eq!(m.chat.input(), "hi");
    }

    #[test]
    fn typing_blocked_during_stream() {
        let mut m = model();
        m.status = Status::streaming();
        let _ = update(&mut m, key(KeyCode::Char('a'), KeyModifiers::NONE));
        assert_eq!(m.chat.input(), "");
    }

    #[test]
    fn enter_on_nonempty_input_spawns() {
        let mut m = model();
        let _ = update(&mut m, key(KeyCode::Char('h'), KeyModifiers::NONE));
        let cmd = update(&mut m, key(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(m.status.kind, StatusKind::Streaming);
        assert!(matches!(cmd, Cmd::Spawn { .. }));
    }

    #[test]
    fn enter_on_empty_input_noop() {
        let mut m = model();
        let cmd = update(&mut m, key(KeyCode::Enter, KeyModifiers::NONE));
        assert!(matches!(cmd, Cmd::None));
        assert_eq!(m.status.kind, StatusKind::Idle);
    }

    #[test]
    fn ctrl_l_clears_history_when_idle() {
        let mut m = model();
        let _ = update(&mut m, key(KeyCode::Char('h'), KeyModifiers::NONE));
        let _ = update(&mut m, key(KeyCode::Enter, KeyModifiers::NONE));
        m.chat.push_assistant("reply".into());
        m.streaming = None;
        m.status = Status::idle("ready");

        let cmd = update(&mut m, key(KeyCode::Char('l'), KeyModifiers::CONTROL));
        assert!(m.chat.history().is_empty());
        assert_eq!(m.status.kind, StatusKind::Notice);
        assert!(matches!(cmd, Cmd::None));
    }

    #[test]
    fn ctrl_l_blocked_during_stream() {
        let mut m = model();
        let _ = update(&mut m, key(KeyCode::Char('h'), KeyModifiers::NONE));
        let _ = update(&mut m, key(KeyCode::Enter, KeyModifiers::NONE));
        let before = m.chat.history().len();

        let _ = update(&mut m, key(KeyCode::Char('l'), KeyModifiers::CONTROL));
        assert_eq!(m.chat.history().len(), before);
    }

    #[test]
    fn first_frame_starts_streaming_state() {
        let mut m = model();
        let _ = update(&mut m, Msg::Frame(frame(1, &[0, 1, 2])));
        assert!(m.streaming.is_some());
        assert_eq!(m.streaming.as_ref().unwrap().latest.step, 1);
    }

    #[test]
    fn later_frames_update_latest_without_resetting_start() {
        let mut m = model();
        let _ = update(&mut m, Msg::Frame(frame(1, &[0, 1, 2])));
        let started = m.streaming.as_ref().unwrap().started;
        let _ = update(&mut m, Msg::Frame(frame(2, &[0])));
        let state = m.streaming.as_ref().unwrap();
        assert_eq!(state.latest.step, 2);
        assert_eq!(state.started, started);
    }

    #[test]
    fn done_with_body_commits_assistant_turn() {
        let mut m = model();
        let _ = update(&mut m, key(KeyCode::Char('h'), KeyModifiers::NONE));
        let _ = update(&mut m, key(KeyCode::Enter, KeyModifiers::NONE));
        let _ = update(&mut m, Msg::Frame(frame(32, &[])));
        let _ = update(&mut m, Msg::StreamDone);
        assert_eq!(m.chat.history().len(), 2);
        assert_eq!(m.status.kind, StatusKind::Idle);
        assert!(m.streaming.is_none());
    }

    #[test]
    fn done_with_empty_body_rolls_back_user_turn() {
        let mut m = model();
        let _ = update(&mut m, key(KeyCode::Char('h'), KeyModifiers::NONE));
        let _ = update(&mut m, key(KeyCode::Enter, KeyModifiers::NONE));
        let empty = StreamFrame {
            step: 0,
            total: NonZeroU64::new(32).unwrap(),
            text: "<|assistant_start|><|assistant_end|>".into(),
            tokens: vec![],
            mask_positions: vec![],
        };
        let _ = update(&mut m, Msg::Frame(empty));
        let _ = update(&mut m, Msg::StreamDone);
        assert!(m.chat.history().is_empty());
        assert_eq!(m.status.kind, StatusKind::Notice);
    }

    #[test]
    fn stream_failed_rolls_back_and_records_error() {
        let mut m = model();
        let _ = update(&mut m, key(KeyCode::Char('h'), KeyModifiers::NONE));
        let _ = update(&mut m, key(KeyCode::Enter, KeyModifiers::NONE));
        let cmd = update(
            &mut m,
            Msg::StreamFailed(ClientError::Transport("boom".into())),
        );
        assert!(m.chat.history().is_empty());
        assert_eq!(m.status.kind, StatusKind::Error);
        assert!(matches!(cmd, Cmd::AbortStream));
    }

    #[test]
    fn tick_increments_monotonically() {
        let mut m = model();
        for _ in 0..5 {
            let _ = update(&mut m, Msg::Tick);
        }
        assert_eq!(m.tick, 5);
    }

    #[test]
    fn send_guards_against_max_length_overflow() {
        let mut m = Model::new(
            "http://x".into(),
            SampleOptions {
                max_length: NonZeroU64::new(5),
                ..SampleOptions::default()
            },
            None,
        );
        for c in "hello world this is a long prompt".chars() {
            let _ = update(
                &mut m,
                Msg::Key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)),
            );
        }
        let cmd = update(&mut m, key(KeyCode::Enter, KeyModifiers::NONE));
        assert!(matches!(cmd, Cmd::None));
        assert_eq!(m.status.kind, StatusKind::Error);
    }
}
