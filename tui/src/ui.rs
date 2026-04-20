use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph, Widget, Wrap},
};

use crate::{
    protocol::{Message, Role, StreamFrame},
    render,
};

pub struct ChatPane<'a> {
    pub history: &'a [Message],
    pub streaming: Option<&'a StreamFrame>,
}

impl Widget for ChatPane<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let mut lines: Vec<Line> = Vec::new();
        for msg in self.history {
            lines.push(role_label(msg.role));
            lines.extend(msg.content.lines().map(|l| Line::from(l.to_string())));
            lines.push(Line::from(""));
        }
        if let Some(frame) = self.streaming {
            lines.push(role_label(Role::Assistant));
            lines.extend(render::render_body(render::extract_assistant(&frame.text)));
        }
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(Block::bordered().title(" chat "))
            .render(area, buf);
    }
}

pub struct InputPane<'a> {
    pub buffer: &'a str,
    pub locked: bool,
}

impl Widget for InputPane<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let style = if self.locked {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default()
        };
        Paragraph::new(self.buffer)
            .style(style)
            .block(Block::bordered().title(" message "))
            .render(area, buf);
    }
}

pub struct StatusBar<'a> {
    pub status: &'a str,
    pub progress: Option<&'a StreamFrame>,
}

impl Widget for StatusBar<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let mut spans: Vec<Span> = vec![Span::styled(
            format!(" {} ", self.status),
            Style::default().add_modifier(Modifier::DIM),
        )];
        if let Some(frame) = self.progress {
            let masks = frame.mask_positions.len();
            let total = u64::from(frame.total);
            spans.push(Span::raw(format!(
                " step {}/{} — {masks} masked ",
                frame.step, total
            )));
        }
        Paragraph::new(Line::from(spans)).render(area, buf);
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
