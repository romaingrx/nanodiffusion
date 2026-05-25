//! Messages + commands — the intents driving the reducer and the effects it produces.
//!
//! * `Msg` is every external thing that can happen: a key press, a tick, a
//!   frame from the network. Anything the outside world throws at us becomes a
//!   `Msg` variant before it touches state.
//! * `Cmd` is every side effect we can ask the runtime to perform. Returning
//!   `Cmd::None` means "no effect"; returning `Cmd::Spawn` is a declarative
//!   request for I/O. The runtime is the only place that knows how to execute
//!   these — `update` just says *what* should happen.
//!
//! This split is what makes `update` a pure function and the codebase
//! testable without tokio.

use crossterm::event::KeyEvent;

use crate::{
    client::ClientError,
    protocol::{ChatRequest, StreamFrame},
};

/// External inputs to the reducer. Rich variants carry their own data so the
/// reducer never has to reach back into live sources.
#[derive(Debug)]
pub enum Msg {
    /// 50 ms heartbeat; drives animation timing and spinner advancement.
    Tick,
    /// A key press from crossterm.
    Key(KeyEvent),
    /// One decoded SSE frame.
    Frame(StreamFrame),
    /// Server closed the stream cleanly (`event: done` or connection EOF).
    StreamDone,
    /// Stream failed — typed so the UI can react differently per kind.
    StreamFailed(ClientError),
    /// crossterm's event stream yielded `None` — terminal is gone.
    TerminalClosed,
}

/// Side effects the reducer requests. The runtime matches on these and runs the
/// actual I/O.
///
/// `#[must_use]` forces callers to thread the value back out of `update` — a
/// silent `Cmd::Spawn` drop would mean "stream silently never started."
#[must_use = "Cmd must be returned to the runtime; dropping it skips the side effect"]
#[derive(Debug)]
pub enum Cmd {
    None,
    /// Start an SSE stream against `url` with this request. The runtime spawns
    /// the tokio task and wires its output back into `Msg::Frame` / `StreamDone`
    /// / `StreamFailed`.
    Spawn {
        url: String,
        req: ChatRequest,
    },
    /// Abort the in-flight stream (drops the session, which aborts the task).
    AbortStream,
    /// Exit the event loop.
    Quit,
}
