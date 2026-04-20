use std::num::NonZeroU64;

use crate::protocol::{ChatRequest, Message, Role};

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

    /// Drain the input into a user turn and build the request to send.
    pub fn commit_user_turn(&mut self, opts: &SampleOptions) -> ChatRequest {
        let content = std::mem::take(&mut self.input);
        self.history.push(Message {
            role: Role::User,
            content,
        });
        ChatRequest {
            messages: self.history.clone(),
            max_length: opts.max_length,
            seed: opts.seed,
            steps: opts.steps,
            temperature: opts.temperature,
            top_k: opts.top_k,
            top_p: opts.top_p,
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_drains_input_and_appends_user_turn() {
        let mut state = ChatState::default();
        state.push_char('h');
        state.push_char('i');
        let req = state.commit_user_turn(&SampleOptions::default());
        assert!(state.input().is_empty());
        assert_eq!(state.history().len(), 1);
        assert!(matches!(state.history()[0].role, Role::User));
        assert_eq!(state.history()[0].content, "hi");
        assert_eq!(req.messages.len(), 1);
        assert!(req.steps.is_none());
    }

    #[test]
    fn commit_threads_sample_options_into_request() {
        let opts = SampleOptions {
            steps: NonZeroU64::new(16),
            temperature: Some(0.7),
            seed: Some(42),
            ..SampleOptions::default()
        };
        let mut state = ChatState::default();
        state.push_char('q');
        let req = state.commit_user_turn(&opts);
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
        let _ = state.commit_user_turn(&SampleOptions::default());
        assert_eq!(state.history().len(), 1);

        assert!(state.rollback_last_user());
        assert_eq!(state.history().len(), 0);
        assert_eq!(state.input(), "hello");
    }

    #[test]
    fn rollback_is_noop_after_assistant_confirms() {
        let mut state = ChatState::default();
        state.push_char('q');
        let _ = state.commit_user_turn(&SampleOptions::default());
        state.push_assistant("answer".into());

        assert!(!state.rollback_last_user());
        assert_eq!(state.history().len(), 2);
    }
}
