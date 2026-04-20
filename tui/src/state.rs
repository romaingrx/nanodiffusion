use crate::protocol::{ChatRequest, Message, Role};

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
    pub fn commit_user_turn(&mut self) -> ChatRequest {
        let content = std::mem::take(&mut self.input);
        self.history.push(Message {
            role: Role::User,
            content,
        });
        ChatRequest {
            messages: self.history.clone(),
            max_length: None,
            seed: None,
            steps: None,
            temperature: None,
            top_k: None,
            top_p: None,
        }
    }

    pub fn push_assistant(&mut self, content: String) {
        self.history.push(Message {
            role: Role::Assistant,
            content,
        });
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
        let req = state.commit_user_turn();
        assert!(state.input().is_empty());
        assert_eq!(state.history().len(), 1);
        assert!(matches!(state.history()[0].role, Role::User));
        assert_eq!(state.history()[0].content, "hi");
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn empty_whitespace_is_treated_as_empty_input() {
        let mut state = ChatState::default();
        state.push_char(' ');
        state.push_char('\t');
        assert!(state.input_is_empty());
    }
}
