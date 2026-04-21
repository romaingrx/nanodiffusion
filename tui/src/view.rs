//! Pure render — the `V` in MVU.
//!
//! `view` takes an immutable [`Model`] and paints it into a ratatui frame. It
//! returns the `Rect` of the chat area so the runtime can overlay effects
//! (see [`crate::runtime`]'s `Reveal::apply` call).
//!
//! Widgets live in [`crate::ui`]; this file only composes them.

use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
};

use crate::{
    model::{Model, StreamingState},
    ui::{ChatPane, Header, InputPane, StatusBar},
};

pub fn view(m: &Model, f: &mut Frame<'_>) -> Rect {
    let [header_area, chat_area, input_area, status_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(3),
        Constraint::Length(3),
        Constraint::Length(1),
    ])
    .areas(f.area());

    f.render_widget(
        Header {
            url: &m.base_url,
            health: m.health.as_ref(),
        },
        header_area,
    );
    f.render_widget(
        ChatPane {
            history: m.chat.history(),
            streaming: m.streaming.as_ref().map(|s| &s.latest),
        },
        chat_area,
    );
    f.render_widget(
        InputPane {
            buffer: m.chat.input(),
            locked: m.is_streaming(),
        },
        input_area,
    );
    f.render_widget(
        StatusBar {
            status: &m.status,
            progress: m.streaming.as_ref().map(|s| &s.latest),
            tok_per_sec: m
                .streaming
                .as_ref()
                .and_then(StreamingState::tokens_per_second),
            prompt_estimate: m.prompt_estimate(),
            tick: m.tick,
        },
        status_area,
    );

    chat_area
}
