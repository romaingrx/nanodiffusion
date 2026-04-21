use std::num::NonZeroU64;

use crate::protocol::{ChatRequest, HealthResponse, Message, Role};

/// Sampling knobs applied uniformly to every request this session makes.
/// Each `None` lets the server use its configured default.
#[derive(Debug, Default, Clone)]
pub struct SampleOptions {
    pub steps: Option<NonZeroU64>,
    pub temperature: Option<f64>,
    pub top_k: Option<u64>,
    pub top_p: Option<f64>,
    pub max_length: Option<NonZeroU64>,
    pub seed: Option<i64>,
}

impl SampleOptions {
    /// Resolve the effective max-length budget: CLI flag overrides server default.
    /// Returns `None` only when neither source knows the budget (health failed
    /// AND no `--max-length` flag was passed).
    pub fn resolved_max_length(&self, health: Option<&HealthResponse>) -> Option<u64> {
        self.max_length
            .map(u64::from)
            .or_else(|| health.map(|h| u64::from(h.sample_defaults.max_length)))
    }

    /// Build a [`ChatRequest`] by snapshotting `messages` and folding in every
    /// sampling knob. Owning the construction here means new knobs added to
    /// `SampleOptions` only require a single edit site.
    pub const fn build_request(&self, messages: Vec<Message>) -> ChatRequest {
        ChatRequest {
            messages,
            max_length: self.max_length,
            seed: self.seed,
            steps: self.steps,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
        }
    }
}

/// Pure conversation state: history + current input buffer.
///
/// Holds no I/O and no timing — the event loop composes this with a [`Session`]
/// to turn keypresses and stream frames into history updates.
///
/// [`Session`]: crate::session::Session
#[derive(Default)]
pub struct ChatState {
    history: Vec<Message>,
    input: String,
}

impl ChatState {
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    pub fn input(&self) -> &str {
        &self.input
    }

    pub fn input_is_empty(&self) -> bool {
        self.input.trim().is_empty()
    }

    pub fn push_char(&mut self, c: char) {
        self.input.push(c);
    }

    pub fn pop_char(&mut self) {
        self.input.pop();
    }

    /// Drain the input buffer into a user message appended to history.
    /// Pair with [`SampleOptions::build_request`] to produce the outgoing
    /// `ChatRequest` for the reducer's `Cmd::Spawn`.
    pub fn commit_user_turn(&mut self) {
        let content = std::mem::take(&mut self.input);
        self.history.push(Message {
            role: Role::User,
            content,
        });
    }

    pub fn push_assistant(&mut self, content: String) {
        self.history.push(Message {
            role: Role::Assistant,
            content,
        });
    }

    /// Undo the most recent user turn and restore its text to the input buffer.
    /// No-op if the last message isn't a user turn (e.g., already confirmed).
    /// Returns whether a rollback actually happened.
    pub fn rollback_last_user(&mut self) -> bool {
        if !matches!(self.history.last().map(|m| m.role), Some(Role::User)) {
            return false;
        }
        let popped = self.history.pop().expect("matches! above guaranteed Some");
        if self.input.is_empty() {
            self.input = popped.content;
        }
        true
    }

    /// Drop the entire conversation history. Input buffer is kept so mid-typed
    /// text isn't lost on a conversation reset.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Conservative GPT-2 BPE estimate for the next request's prompt length.
    /// Uses ~3.5 chars/token on English plus a small framing budget per turn —
    /// close enough to pre-flight against the server's `max_length` without
    /// round-tripping the actual tokenizer.
    pub fn estimate_prompt_tokens(&self) -> u64 {
        const PREAMBLE: u64 = 2;
        const FRAMING_PER_TURN: u64 = 2;
        PREAMBLE
            + self
                .history
                .iter()
                .map(|m| FRAMING_PER_TURN + chars_to_tokens(&m.content))
                .sum::<u64>()
    }
}

fn chars_to_tokens(text: &str) -> u64 {
    let chars = text.chars().count() as u64;
    (chars * 2).div_ceil(7)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_drains_input_and_appends_user_turn() {
        let mut state = ChatState::default();
        state.push_char('h');
        state.push_char('i');
        state.commit_user_turn();
        assert!(state.input().is_empty());
        assert_eq!(state.history().len(), 1);
        assert!(matches!(state.history()[0].role, Role::User));
        assert_eq!(state.history()[0].content, "hi");
    }

    #[test]
    fn build_request_snapshots_history_and_threads_sample_options() {
        let opts = SampleOptions {
            steps: NonZeroU64::new(16),
            temperature: Some(0.7),
            seed: Some(42),
            ..SampleOptions::default()
        };
        let mut state = ChatState::default();
        state.push_char('q');
        state.commit_user_turn();
        let req = opts.build_request(state.history().to_vec());
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.steps, NonZeroU64::new(16));
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.seed, Some(42));
    }

    #[test]
    fn empty_whitespace_is_treated_as_empty_input() {
        let mut state = ChatState::default();
        state.push_char(' ');
        state.push_char('\t');
        assert!(state.input_is_empty());
    }

    #[test]
    fn rollback_pops_user_and_restores_input_buffer() {
        let mut state = ChatState::default();
        for c in "hello".chars() {
            state.push_char(c);
        }
        state.commit_user_turn();
        assert_eq!(state.history().len(), 1);

        assert!(state.rollback_last_user());
        assert_eq!(state.history().len(), 0);
        assert_eq!(state.input(), "hello");
    }

    #[test]
    fn rollback_is_noop_after_assistant_confirms() {
        let mut state = ChatState::default();
        state.push_char('q');
        state.commit_user_turn();
        state.push_assistant("answer".into());

        assert!(!state.rollback_last_user());
        assert_eq!(state.history().len(), 2);
    }

    #[test]
    fn clear_history_drops_messages_preserves_input() {
        let mut state = ChatState::default();
        state.push_char('q');
        state.commit_user_turn();
        state.push_assistant("answer".into());
        for c in "next".chars() {
            state.push_char(c);
        }

        state.clear_history();
        assert_eq!(state.history().len(), 0);
        assert_eq!(state.input(), "next");
    }

    #[test]
    fn estimate_grows_with_history() {
        let mut state = ChatState::default();
        let empty = state.estimate_prompt_tokens();
        for c in "hello".chars() {
            state.push_char(c);
        }
        state.commit_user_turn();
        assert!(state.estimate_prompt_tokens() > empty);
    }
}
