//! Pure application state — the `M` in MVU.
//!
//! Everything here is plain data. No tasks, no channels, no tokio types.
//! Transitions happen exclusively in [`crate::update`]; effects in
//! [`crate::runtime`]. Keeping this module free of I/O means every state
//! transition can be unit-tested by constructing a `Model` in memory and
//! calling `update(&mut m, msg)`.

use std::time::Instant;

use crate::{
    protocol::{HealthResponse, StreamFrame},
    state::{ChatState, SampleOptions},
};

pub struct Model {
    pub base_url: String,
    pub health: Option<HealthResponse>,
    pub sample_opts: SampleOptions,
    pub chat: ChatState,
    pub status: Status,
    pub tick: u64,
    pub streaming: Option<StreamingState>,
    pub should_quit: bool,
}

impl Model {
    pub fn new(
        base_url: String,
        sample_opts: SampleOptions,
        health: Option<HealthResponse>,
    ) -> Self {
        let hint = if health.is_some() {
            "ready · enter to send · ctrl-l clears · esc cancels · ctrl-c quits"
        } else {
            "ready (server health unknown) · enter to send · ctrl-c quits"
        };
        Self {
            base_url,
            health,
            sample_opts,
            chat: ChatState::default(),
            status: Status::idle(hint),
            tick: 0,
            streaming: None,
            should_quit: false,
        }
    }

    /// True from the moment the user commits a turn until the stream ends
    /// (clean, cancelled, or failed). This is tied to `status.kind` rather
    /// than `streaming.is_some()` so the "request sent but no frame yet" gap
    /// still blocks input.
    pub const fn is_streaming(&self) -> bool {
        matches!(self.status.kind, StatusKind::Streaming)
    }

    pub fn prompt_estimate(&self) -> Option<PromptEstimate> {
        let max = self.sample_opts.resolved_max_length(self.health.as_ref())?;
        Some(PromptEstimate {
            used: self.chat.estimate_prompt_tokens(),
            max,
        })
    }
}

/// State of the in-flight response. Present iff at least one frame has arrived
/// from the server. Before the first frame the UI relies on `Status::Streaming`
/// to show the "streaming…" indicator; once frames arrive, `streaming` drives
/// the mask overlay and tok/s readout.
pub struct StreamingState {
    pub latest: StreamFrame,
    pub started: Instant,
}

impl StreamingState {
    /// Throughput based on revealed positions so far. Returns `None` before
    /// enough wall time has elapsed to be meaningful.
    ///
    /// f64 precision is fine — counts are bounded by `max_seq_len` (thousands),
    /// nowhere near `2^52`.
    #[allow(clippy::cast_precision_loss)]
    pub fn tokens_per_second(&self) -> Option<f64> {
        const WARMUP: f64 = 0.05;
        let total = u64::from(self.latest.total) as f64;
        let revealed = total - self.latest.mask_positions.len() as f64;
        let elapsed = self.started.elapsed().as_secs_f64();
        (elapsed > WARMUP).then_some(revealed / elapsed)
    }
}

/// Status line. `kind` drives the icon/colour; `text` is the message. Encoding
/// them together means there's no "unit variant has no text" special case.
#[derive(Clone)]
pub struct Status {
    pub kind: StatusKind,
    pub text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatusKind {
    Idle,
    Streaming,
    Error,
    Notice,
}

impl Status {
    pub fn idle(text: impl Into<String>) -> Self {
        Self {
            kind: StatusKind::Idle,
            text: text.into(),
        }
    }
    pub fn streaming() -> Self {
        Self {
            kind: StatusKind::Streaming,
            text: "streaming".into(),
        }
    }
    pub fn error(text: impl Into<String>) -> Self {
        Self {
            kind: StatusKind::Error,
            text: text.into(),
        }
    }
    pub fn notice(text: impl Into<String>) -> Self {
        Self {
            kind: StatusKind::Notice,
            text: text.into(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct PromptEstimate {
    pub used: u64,
    pub max: u64,
}

impl PromptEstimate {
    /// Fraction used as a percentage, saturating on overflow, `0` when `max == 0`.
    pub const fn used_pct(self) -> u64 {
        match self.used.saturating_mul(100).checked_div(self.max) {
            Some(pct) => pct,
            None => 0,
        }
    }

    /// Whether the prompt occupies more than `threshold_pct` of the budget.
    pub const fn is_near_limit(self, threshold_pct: u64) -> bool {
        self.used_pct() >= threshold_pct
    }
}
