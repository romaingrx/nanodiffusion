use std::time::Instant;

use tokio::{sync::mpsc, task::AbortHandle};

use crate::{
    client::{ClientEvent, stream_chat},
    protocol::{ChatRequest, StreamFrame},
};

/// One in-flight chat completion: the receiver side of the stream, plus the
/// abort handle for the task feeding it. Dropping the session aborts the task.
pub struct Session {
    rx: mpsc::Receiver<ClientEvent>,
    abort: AbortHandle,
    latest: Option<StreamFrame>,
    started: Instant,
}

/// Coarse update the app reacts to, decoupled from the wire [`ClientEvent`].
pub enum SessionUpdate {
    Frame,
    Done,
    Error(String),
}

impl Session {
    pub fn spawn(base_url: String, request: ChatRequest) -> Self {
        let (tx, rx) = mpsc::channel(64);
        let handle = tokio::spawn(async move {
            let _ = stream_chat(&base_url, &request, tx).await;
        });
        Self {
            rx,
            abort: handle.abort_handle(),
            latest: None,
            started: Instant::now(),
        }
    }

    pub fn latest(&self) -> Option<&StreamFrame> {
        self.latest.as_ref()
    }

    pub fn take_latest(&mut self) -> Option<StreamFrame> {
        self.latest.take()
    }

    /// Throughput in tokens-per-second based on revealed positions so far.
    /// Returns `None` before enough wall time has elapsed to be meaningful.
    pub fn tokens_per_second(&self) -> Option<f64> {
        const WARMUP: f64 = 0.05;
        let frame = self.latest.as_ref()?;
        let total = u64::from(frame.total) as f64;
        let revealed = total - frame.mask_positions.len() as f64;
        let elapsed = self.started.elapsed().as_secs_f64();
        (elapsed > WARMUP).then_some(revealed / elapsed)
    }

    /// Await the next wire event and fold it into local state.
    /// Returns `None` when the stream channel closes.
    pub async fn poll(&mut self) -> Option<SessionUpdate> {
        match self.rx.recv().await {
            Some(ClientEvent::Frame(frame)) => {
                self.latest = Some(frame);
                Some(SessionUpdate::Frame)
            }
            Some(ClientEvent::Done) => Some(SessionUpdate::Done),
            Some(ClientEvent::Error(e)) => Some(SessionUpdate::Error(e)),
            None => None,
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.abort.abort();
    }
}
