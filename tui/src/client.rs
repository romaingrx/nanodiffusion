use std::time::Duration;

use color_eyre::{
    Result,
    eyre::{Context, eyre},
};
use futures::StreamExt;
use reqwest_eventsource::{Error as SseError, Event, RequestBuilderExt, retry};
use tokio::sync::mpsc;

use crate::{
    msg::Msg,
    protocol::{ChatRequest, HealthResponse, StreamFrame},
};

const STREAM_PATH: &str = "/api/chat/stream";
const HEALTH_PATH: &str = "/api/health";
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const READ_TIMEOUT: Duration = Duration::from_mins(5);
const HEALTH_TIMEOUT: Duration = Duration::from_secs(3);

/// Typed error surface for the SSE client, so the UI can distinguish between
/// "server said no" (actionable: fix the request), "transport failed" (retry?),
/// and "we couldn't decode a frame" (bug / version skew).
#[derive(Debug, Clone)]
pub enum ClientError {
    Rejected { status: u16, detail: String },
    Transport(String),
    Decode(String),
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rejected { status, detail } => write!(f, "server {status}: {detail}"),
            Self::Transport(msg) => write!(f, "transport: {msg}"),
            Self::Decode(msg) => write!(f, "decode: {msg}"),
        }
    }
}

/// Build the shared HTTP client used for both health probes and SSE streams.
///
/// Timeouts are set for the long path (SSE: 5 s connect, 5 min read); short
/// calls like `fetch_health` apply a per-request override via `.timeout(...)`.
/// `reqwest::Client` is internally `Arc`-wrapped, so cloning this value is a
/// refcount bump — share it freely.
pub fn build_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(READ_TIMEOUT)
        .build()
        .wrap_err("build http client")
}

pub async fn fetch_health(http_client: &reqwest::Client, base_url: &str) -> Result<HealthResponse> {
    let url = join(base_url, HEALTH_PATH);
    let res = http_client
        .get(&url)
        .timeout(HEALTH_TIMEOUT)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;

    let status = res.status();
    if !status.is_success() {
        return Err(eyre!("health returned {status}"));
    }
    res.json::<HealthResponse>()
        .await
        .wrap_err("decode health response")
}

pub async fn stream_chat(
    http_client: &reqwest::Client,
    base_url: &str,
    req: &ChatRequest,
    tx: mpsc::Sender<Msg>,
) -> Result<()> {
    let url = join(base_url, STREAM_PATH);
    tracing::debug!(url = %url, turns = req.messages.len(), "opening sse stream");

    let mut es = http_client
        .post(&url)
        .header("accept", "text/event-stream")
        .json(req)
        .eventsource()
        .wrap_err("build sse request")?;
    es.set_retry_policy(Box::new(retry::Never));

    while let Some(event) = es.next().await {
        match event {
            Ok(Event::Open) => {}
            Ok(Event::Message(sse)) => match serde_json::from_str::<StreamFrame>(&sse.data) {
                Ok(frame) => {
                    if tx.send(Msg::Frame(frame)).await.is_err() {
                        return Ok(());
                    }
                }
                Err(e) => {
                    tracing::error!(error = %e, data = %sse.data, "failed to decode frame");
                    let _ = tx
                        .send(Msg::StreamFailed(ClientError::Decode(e.to_string())))
                        .await;
                    return Ok(());
                }
            },
            Err(SseError::StreamEnded) => {
                let _ = tx.send(Msg::StreamDone).await;
                return Ok(());
            }
            Err(SseError::InvalidStatusCode(status, resp)) => {
                let detail = extract_detail(resp).await;
                tracing::warn!(%status, %detail, "server rejected request");
                let err = ClientError::Rejected {
                    status: status.as_u16(),
                    detail,
                };
                let _ = tx.send(Msg::StreamFailed(err)).await;
                return Ok(());
            }
            Err(e) => {
                tracing::error!(error = %e, "sse stream error");
                let _ = tx
                    .send(Msg::StreamFailed(ClientError::Transport(e.to_string())))
                    .await;
                return Ok(());
            }
        }
    }
    let _ = tx.send(Msg::StreamDone).await;
    Ok(())
}

fn join(base: &str, path: &str) -> String {
    format!("{}{path}", base.trim_end_matches('/'))
}

/// Extract `FastAPI`'s `{"detail": ...}` string when present; fall back to the
/// raw body decoded as UTF-8.
async fn extract_detail(resp: reqwest::Response) -> String {
    let bytes = resp.bytes().await.unwrap_or_default();
    serde_json::from_slice::<serde_json::Value>(&bytes)
        .ok()
        .and_then(|v| v.get("detail").and_then(|d| d.as_str()).map(str::to_string))
        .unwrap_or_else(|| String::from_utf8_lossy(&bytes).trim().to_string())
}
