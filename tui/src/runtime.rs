//! The impure shell. Translates async I/O to [`Msg`], calls `update`, executes
//! [`Cmd`], renders [`view`]. The only file in the crate that speaks tokio.
//!
//! Splitting responsibilities this way (pure model/update/view + impure
//! runtime) makes every state transition testable without a tokio runtime
//! and keeps the side-effect surface area small and auditable.

use std::time::Duration;

use color_eyre::Result;
use crossterm::event::{Event, EventStream};
use futures::StreamExt;
use tokio::time::interval;

use crate::{
    effects::Reveal,
    model::Model,
    msg::{Cmd, Msg},
    session::Session,
    terminal::Tui,
    update::update,
    view::view,
};

const TICK: Duration = Duration::from_millis(50);

/// Drive the event loop until `Cmd::Quit` or the terminal closes.
///
/// One source of truth per concern:
/// - `model`: pure application state
/// - `session`: the in-flight tokio task's abort handle + frame channel
/// - `reveal`: view-side animation state (tachyonfx's `EffectManager` and
///   tick bookkeeping — purposefully not in `Model`)
#[allow(clippy::future_not_send)] // current_thread runtime
pub async fn run(mut model: Model, term: &mut Tui, http_client: reqwest::Client) -> Result<()> {
    let mut events = EventStream::new();
    let mut ticker = interval(TICK);
    let mut session: Option<Session> = None;
    let mut reveal = Reveal::new();

    while !model.should_quit {
        term.draw(|f| {
            let chat_area = view(&model, f);
            reveal.apply(f.buffer_mut(), chat_area);
        })?;

        let msg = tokio::select! {
            _ = ticker.tick() => Msg::Tick,
            maybe_evt = events.next() => crossterm_msg(maybe_evt)?,
            // channel closed: treat as a clean end to stay aligned with `StreamDone`
            maybe_msg = poll_session(session.as_mut()) => maybe_msg.unwrap_or(Msg::StreamDone),
        };

        // View-side reaction before update so the pulse lands on the same frame.
        if let Msg::Frame(ref f) = msg {
            reveal.observe(f);
        }

        let cmd = update(&mut model, msg);
        execute(cmd, &http_client, &mut session, &mut reveal);
    }
    Ok(())
}

fn execute(
    cmd: Cmd,
    http_client: &reqwest::Client,
    session: &mut Option<Session>,
    reveal: &mut Reveal,
) {
    match cmd {
        // `Cmd::Quit` is already reflected in `model.should_quit` — the loop
        // guard picks it up on next iteration. Dropping `session` on function
        // return cleans up the in-flight task.
        Cmd::None | Cmd::Quit => {}
        Cmd::Spawn { url, req } => {
            // dropping the previous Session aborts its task via Drop.
            // Client::clone is a cheap internal refcount bump.
            *session = Some(Session::spawn(http_client.clone(), url, req));
            reveal.reset();
        }
        Cmd::AbortStream => {
            *session = None;
            reveal.reset();
        }
    }
}

/// Await the next message from the in-flight session (if any). If no session
/// is live, pend forever so the `select!` arm stays inert.
async fn poll_session(session: Option<&mut Session>) -> Option<Msg> {
    match session {
        Some(s) => s.poll().await,
        None => std::future::pending().await,
    }
}

fn crossterm_msg(evt: Option<std::io::Result<Event>>) -> Result<Msg> {
    match evt {
        Some(Ok(Event::Key(key))) => Ok(Msg::Key(key)),
        Some(Ok(_)) => Ok(Msg::Tick), // resize/focus/paste: treat as a tick to force redraw
        Some(Err(e)) => Err(e.into()),
        None => Ok(Msg::TerminalClosed),
    }
}
