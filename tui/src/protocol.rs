//! Wire types regenerated from `schemas/protocol.json` on every build.
//!
//! The Python side owns the schema; this macro keeps Rust in sync with zero
//! drift. If compilation fails after editing Pydantic models, rebuild in
//! Python first: `just schema`.

#![allow(clippy::all)]
#![allow(warnings)]

typify::import_types!(schema = "../schemas/protocol.json",);

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use super::*;

    #[test]
    fn stream_frame_roundtrips() {
        let raw = r#"{
            "step": 3,
            "total": 32,
            "tokens": [1, 2, 4],
            "text": "hi",
            "mask_positions": [0, 5, 7]
        }"#;
        let frame: StreamFrame = serde_json::from_str(raw).unwrap();
        assert_eq!(frame.step, 3);
        assert_eq!(frame.total, NonZeroU64::new(32).unwrap());
        assert_eq!(frame.mask_positions, vec![0, 5, 7]);
    }

    #[test]
    fn chat_request_serializes_without_optionals() {
        let req = ChatRequest {
            messages: vec![Message {
                role: Role::User,
                content: "hi".into(),
            }],
            max_length: None,
            seed: None,
            steps: None,
            temperature: None,
            top_k: None,
            top_p: None,
        };
        let s = serde_json::to_string(&req).unwrap();
        assert!(s.contains("\"role\":\"user\""));
        assert!(s.contains("\"content\":\"hi\""));
        assert!(!s.contains("temperature"));
    }
}
