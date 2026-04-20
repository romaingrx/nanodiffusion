use ratatui::{
    style::{Color, Style},
    text::{Line, Span},
};

const MASK_LITERAL: &str = "<|mask|>";
const ASSISTANT_START: &str = "<|assistant_start|>";
const ASSISTANT_END: &str = "<|assistant_end|>";
const EOS: &str = "<|eos|>";

const MASK_GLYPH: &str = "░░";

const MASK_COLOR: Color = Color::Rgb(0x44, 0x44, 0x44);
const SETTLED_COLOR: Color = Color::Rgb(0x66, 0x99, 0xCC);

/// Strip prompt prefix and closing markers, keeping just the assistant body.
pub fn extract_assistant(text: &str) -> &str {
    let after_start = text
        .rsplit_once(ASSISTANT_START)
        .map(|(_, rest)| rest)
        .unwrap_or(text);
    let before_end = after_start
        .split_once(ASSISTANT_END)
        .map(|(head, _)| head)
        .unwrap_or(after_start);
    before_end.split_once(EOS).map(|(h, _)| h).unwrap_or(before_end)
}

/// Render a partially-unmasked assistant body as LLaDA-palette spans.
/// Mask literals render as a short dim glyph; decoded text uses the settled color.
pub fn render_body(body: &str) -> Vec<Line<'static>> {
    body.split('\n').map(render_line).collect()
}

fn render_line(line: &str) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut remaining = line;
    while let Some(idx) = remaining.find(MASK_LITERAL) {
        if idx > 0 {
            spans.push(Span::styled(
                remaining[..idx].to_string(),
                Style::default().fg(SETTLED_COLOR),
            ));
        }
        spans.push(Span::styled(
            MASK_GLYPH.to_string(),
            Style::default().fg(MASK_COLOR),
        ));
        remaining = &remaining[idx + MASK_LITERAL.len()..];
    }
    if !remaining.is_empty() {
        spans.push(Span::styled(
            remaining.to_string(),
            Style::default().fg(SETTLED_COLOR),
        ));
    }
    Line::from(spans)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_assistant_body() {
        let raw = "<|bos|><|user_start|>hi<|user_end|><|assistant_start|>hello world<|assistant_end|>";
        assert_eq!(extract_assistant(raw), "hello world");
    }

    #[test]
    fn extract_falls_back_to_whole_string() {
        assert_eq!(extract_assistant("no markers here"), "no markers here");
    }

    #[test]
    fn render_splits_on_mask_literals() {
        let lines = render_body("hello <|mask|><|mask|> world");
        assert_eq!(lines.len(), 1);
        let spans = &lines[0].spans;
        assert_eq!(spans.len(), 4);
        assert_eq!(spans[0].content, "hello ");
        assert_eq!(spans[1].content, MASK_GLYPH);
        assert_eq!(spans[2].content, MASK_GLYPH);
        assert_eq!(spans[3].content, " world");
    }

    #[test]
    fn render_preserves_newlines_as_separate_lines() {
        let lines = render_body("a\nb");
        assert_eq!(lines.len(), 2);
    }
}
