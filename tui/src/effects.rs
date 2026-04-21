use std::time::{Duration, Instant};

use ratatui::{buffer::Buffer, layout::Rect, style::Color};
use tachyonfx::{EffectManager, Interpolation, fx};

use crate::protocol::StreamFrame;

const REVEAL_COLOR: Color = Color::Rgb(0xFF, 0xAA, 0x33);
const REVEAL_DURATION: Duration = Duration::from_millis(180);

/// Triggers a brief reveal animation on the chat area whenever the server
/// unmasks tokens. The animation is driven by tachyonfx's [`EffectManager`],
/// which is ticked each frame with the elapsed wall-clock time.
pub struct Reveal {
    manager: EffectManager<()>,
    last_tick: Instant,
    previous_masks: Option<usize>,
}

impl Reveal {
    pub fn new() -> Self {
        Self {
            manager: EffectManager::default(),
            last_tick: Instant::now(),
            previous_masks: None,
        }
    }

    /// Consume a fresh [`StreamFrame`] and enqueue a reveal effect when the
    /// mask count drops — i.e., the server just filled one or more positions.
    pub fn observe(&mut self, frame: &StreamFrame) {
        let current = frame.mask_positions.len();
        if self.previous_masks.is_some_and(|prev| current < prev) {
            self.manager.add_unique_effect(
                (),
                fx::fade_from_fg(REVEAL_COLOR, (REVEAL_DURATION, Interpolation::QuadOut)),
            );
        }
        self.previous_masks = Some(current);
    }

    /// Clear any state tracked across a streaming session. Call on finalize or cancel.
    pub const fn reset(&mut self) {
        self.previous_masks = None;
    }

    /// Advance effects by real elapsed time and paint them onto `buf` within `area`.
    pub fn apply(&mut self, buf: &mut Buffer, area: Rect) {
        let now = Instant::now();
        let elapsed = now.saturating_duration_since(self.last_tick);
        self.last_tick = now;
        self.manager.process_effects(elapsed, buf, area);
    }
}

impl Default for Reveal {
    fn default() -> Self {
        Self::new()
    }
}
