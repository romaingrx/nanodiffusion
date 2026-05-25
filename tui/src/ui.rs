use ratatui::{
    buffer::Buffer,
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Padding, Paragraph, Widget, Wrap},
};

use crate::{
    model::{PromptEstimate, Status, StatusKind},
    protocol::{HealthResponse, Message, Role, StreamFrame},
    render,
};

const ACCENT: Color = Color::Rgb(0x66, 0x99, 0xCC);
const MUTED: Color = Color::Rgb(0x80, 0x80, 0x80);
const ASSISTANT: Color = Color::Rgb(0xC5, 0x7F, 0xC5);
const ERROR: Color = Color::Rgb(0xE5, 0x6B, 0x6B);
const WARN: Color = Color::Rgb(0xFF, 0xAA, 0x33);
const SUCCESS: Color = Color::Rgb(0x7F, 0xC5, 0x7F);

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const SEP: &str = "│";

const WARN_THRESHOLD_PCT: u64 = 80;

pub struct Header<'a> {
    pub url: &'a str,
    pub health: Option<&'a HealthResponse>,
}

impl Widget for Header<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let title = Span::styled(
            " nanodiffusion ",
            Style::default()
                .fg(Color::Black)
                .bg(ACCENT)
                .add_modifier(Modifier::BOLD),
        );

        let right_spans = self.health.map_or_else(
            || {
                vec![Span::styled(
                    format!(" {} (offline) ", self.url),
                    Style::default().fg(WARN),
                )]
            },
            |h| {
                vec![
                    Span::styled(
                        format!(" step {} ", h.train_step),
                        Style::default().fg(ACCENT),
                    ),
                    sep(),
                    Span::styled(
                        format!(" ctx {} ", u64::from(h.max_seq_len)),
                        Style::default().fg(MUTED),
                    ),
                    sep(),
                    Span::styled(format!(" {} ", self.url), Style::default().fg(MUTED)),
                ]
            },
        );

        Paragraph::new(Line::from(vec![title])).render(area, buf);
        Paragraph::new(Line::from(right_spans))
            .alignment(Alignment::Right)
            .render(area, buf);
    }
}

pub struct ChatPane<'a> {
    pub history: &'a [Message],
    pub streaming: Option<&'a StreamFrame>,
}

impl Widget for ChatPane<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let mut lines: Vec<Line> = Vec::new();
        if self.history.is_empty() && self.streaming.is_none() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "no messages yet — type below and press enter",
                Style::default().fg(MUTED).add_modifier(Modifier::ITALIC),
            )));
        }
        for (i, msg) in self.history.iter().enumerate() {
            if i > 0 {
                lines.push(Line::from(""));
            }
            lines.push(role_label(msg.role));
            lines.extend(msg.content.lines().map(|l| Line::from(format!("  {l}"))));
        }
        if let Some(frame) = self.streaming {
            if !self.history.is_empty() {
                lines.push(Line::from(""));
            }
            lines.push(role_label(Role::Assistant));
            let body = render::extract_assistant(&frame.text);
            for line in render::render_body(body) {
                let mut indented = vec![Span::raw("  ")];
                indented.extend(line.spans);
                lines.push(Line::from(indented));
            }
        }
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(panel(" chat ", MUTED))
            .render(area, buf);
    }
}

pub struct InputPane<'a> {
    pub buffer: &'a str,
    pub locked: bool,
}

impl Widget for InputPane<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let styles = InputStyles::resolve(self.locked);
        let mut spans = vec![
            Span::styled("▸ ", styles.prompt),
            Span::styled(self.buffer.to_string(), styles.text),
        ];
        if !self.locked {
            spans.push(Span::styled(
                "█",
                Style::default()
                    .fg(ACCENT)
                    .add_modifier(Modifier::SLOW_BLINK),
            ));
        }
        Paragraph::new(Line::from(spans))
            .block(panel(" message ", styles.border_color))
            .render(area, buf);
    }
}

/// Style bundle for the input pane. Grouping these in one struct makes the
/// branch-by-branch coupling explicit and keeps the render path linear.
struct InputStyles {
    prompt: Style,
    text: Style,
    border_color: Color,
}

impl InputStyles {
    fn resolve(locked: bool) -> Self {
        if locked {
            Self {
                prompt: Style::default().fg(MUTED),
                text: Style::default().fg(MUTED),
                border_color: MUTED,
            }
        } else {
            Self {
                prompt: Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                text: Style::default().fg(Color::White),
                border_color: ACCENT,
            }
        }
    }
}

pub struct StatusBar<'a> {
    pub status: &'a Status,
    pub progress: Option<&'a StreamFrame>,
    pub tok_per_sec: Option<f64>,
    pub prompt_estimate: Option<PromptEstimate>,
    pub tick: u64,
}

impl Widget for StatusBar<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let (icon, color) = status_badge(self.status, self.tick);

        let left = Line::from(vec![
            Span::raw(" "),
            Span::styled(
                icon,
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(self.status.text.as_str(), Style::default().fg(Color::White)),
        ]);

        let mut right: Vec<Span> = Vec::new();
        if let Some(frame) = self.progress {
            let masks = frame.mask_positions.len();
            let total = u64::from(frame.total);
            push_sep(&mut right, format!("step {}/{total}", frame.step), MUTED);
            push_sep(&mut right, format!("{masks} masked"), MUTED);
            if let Some(rate) = self.tok_per_sec {
                push_sep(&mut right, format!("{rate:.1} tok/s"), MUTED);
            }
        }
        if let Some(est) = self.prompt_estimate {
            let color = if est.is_near_limit(WARN_THRESHOLD_PCT) {
                WARN
            } else {
                MUTED
            };
            push_sep(&mut right, format!("~{}/{} tok", est.used, est.max), color);
        }
        if !right.is_empty() {
            right.push(Span::raw(" "));
        }

        Paragraph::new(left).render(area, buf);
        Paragraph::new(Line::from(right))
            .alignment(Alignment::Right)
            .render(area, buf);
    }
}

const fn status_badge(status: &Status, tick: u64) -> (&'static str, Color) {
    match status.kind {
        StatusKind::Idle => ("●", SUCCESS),
        StatusKind::Streaming => (spinner_frame(tick), ACCENT),
        StatusKind::Error => ("●", ERROR),
        StatusKind::Notice => ("●", WARN),
    }
}

fn push_sep(spans: &mut Vec<Span<'static>>, text: String, color: Color) {
    if !spans.is_empty() {
        spans.push(Span::styled(format!(" {SEP} "), Style::default().fg(MUTED)));
    }
    spans.push(Span::styled(text, Style::default().fg(color)));
}

fn sep() -> Span<'static> {
    Span::styled(SEP, Style::default().fg(MUTED))
}

fn role_label(role: Role) -> Line<'static> {
    let (glyph, label, color) = match role {
        Role::User => ("▸", "you", ACCENT),
        Role::Assistant => ("◆", "model", ASSISTANT),
        Role::System => ("◇", "system", WARN),
    };
    Line::from(vec![Span::styled(
        format!("{glyph} {label}"),
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    )])
}

/// Spinner driven by the event-loop tick count, so the UI has no hidden time
/// dependency (was previously reading `SystemTime::now()` each frame).
const fn spinner_frame(tick: u64) -> &'static str {
    // tick is a 50ms counter; slow it to ~125ms per frame so the glyph isn't a blur.
    let idx = (tick / 2) as usize % SPINNER.len();
    SPINNER[idx]
}

/// Shared chrome for every framed pane: rounded border, 1-col horizontal padding,
/// coloured title. Factored out so widgets don't repeat the builder chain.
fn panel(title: &'static str, border_color: Color) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_color))
        .padding(Padding::horizontal(1))
        .title(title)
}
