//! Thin wrapper around an in-flight SSE task. Owns the abort handle and the
//! receiver side of the frame channel. Dropping a `Session` aborts the task —
//! which is exactly the lifecycle guarantee the runtime wants when it assigns
//! `*session = None` or `*session = Some(new)`.
//!
//! Frame-level state (`latest`, `started`) lives on [`crate::model::Model`]
//! instead of here — the spawned task sends `Msg` values directly through the
//! channel and the reducer folds them into `Model::streaming`.

use tokio::{sync::mpsc, task::AbortHandle};

use crate::{client::stream_chat, msg::Msg, protocol::ChatRequest};

pub struct Session {
    rx: mpsc::Receiver<Msg>,
    abort: AbortHandle,
}

impl Session {
    /// Spawn a task streaming frames into an internal channel.
    ///
    /// `http_client` is the shared `reqwest::Client` (cheap to clone —
    /// internally `Arc`-wrapped). Sharing it avoids a new connection pool + TLS
    /// handshake per turn.
    pub fn spawn(http_client: reqwest::Client, base_url: String, request: ChatRequest) -> Self {
        let (tx, rx) = mpsc::channel(64);
        let handle = tokio::spawn(async move {
            let _ = stream_chat(&http_client, &base_url, &request, tx).await;
        });
        Self {
            rx,
            abort: handle.abort_handle(),
        }
    }

    /// Next message from the stream, or `None` if the channel closed.
    pub async fn poll(&mut self) -> Option<Msg> {
        self.rx.recv().await
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.abort.abort();
    }
}
