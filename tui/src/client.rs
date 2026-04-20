use std::time::Duration;

use anyhow::{Context, Result};
use eventsource_client::{Client, ClientBuilder, Error as SseError, ReconnectOptions, SSE};
use futures::TryStreamExt;
use launchdarkly_sdk_transport::HyperTransport;
use tokio::sync::mpsc;

use crate::protocol::{ChatRequest, StreamFrame};

const STREAM_PATH: &str = "/api/chat/stream";
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const READ_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Debug)]
pub enum ClientEvent {
    Frame(StreamFrame),
    Done,
    Error(String),
}

pub async fn stream_chat(
    base_url: &str,
    req: &ChatRequest,
    tx: mpsc::Sender<ClientEvent>,
) -> Result<()> {
    let client = build_client(base_url, req)?;
    let mut stream = Box::pin(client.stream());

    loop {
        match stream.try_next().await {
            Ok(Some(SSE::Event(ev))) => {
                let frame = serde_json::from_str::<StreamFrame>(&ev.data)
                    .context("decode StreamFrame")
                    .inspect_err(|e| {
                        let _ = tx.try_send(ClientEvent::Error(e.to_string()));
                    })?;
                if tx.send(ClientEvent::Frame(frame)).await.is_err() {
                    return Ok(());
                }
            }
            Ok(Some(_)) => continue,
            Ok(None) | Err(SseError::Eof) => {
                let _ = tx.send(ClientEvent::Done).await;
                return Ok(());
            }
            Err(e) => {
                let message = e.to_string();
                let _ = tx.send(ClientEvent::Error(message)).await;
                return Err(anyhow::Error::msg(e.to_string())).context("sse stream");
            }
        }
    }
}

fn build_client(base_url: &str, req: &ChatRequest) -> Result<impl Client> {
    let url = format!("{}{}", base_url.trim_end_matches('/'), STREAM_PATH);
    let body = serde_json::to_string(req).context("serialize request")?;

    let transport = HyperTransport::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(READ_TIMEOUT)
        .build_http()
        .map_err(|e| anyhow::Error::msg(e.to_string()))
        .context("build http transport")?;

    let client = ClientBuilder::for_url(&url)
        .context("invalid url")?
        .method("POST".into())
        .header("content-type", "application/json")
        .map_err(|e| anyhow::Error::msg(e.to_string()))?
        .header("accept", "text/event-stream")
        .map_err(|e| anyhow::Error::msg(e.to_string()))?
        .body(body)
        .reconnect(ReconnectOptions::reconnect(false).build())
        .build_with_transport(transport);

    Ok(client)
}
