#![allow(dead_code)]

use std::time::Duration;

use anyhow::{Context, Result};
use eventsource_client::{Client, ClientBuilder, ReconnectOptions, SSE};
use futures::TryStreamExt;
use launchdarkly_sdk_transport::HyperTransport;
use tokio::sync::mpsc;

use crate::protocol::{ChatRequest, StreamFrame};

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
    let url = format!("{}/api/chat/stream", base_url.trim_end_matches('/'));
    let body = serde_json::to_string(req).context("serialize request")?;

    let transport = HyperTransport::builder()
        .connect_timeout(Duration::from_secs(5))
        .read_timeout(Duration::from_secs(300))
        .build_http()
        .map_err(|e| anyhow::anyhow!("build transport: {e:?}"))?;

    let client = ClientBuilder::for_url(&url)
        .context("build sse client")?
        .method("POST".into())
        .header("content-type", "application/json")
        .map_err(|e| anyhow::anyhow!("set content-type: {e:?}"))?
        .header("accept", "text/event-stream")
        .map_err(|e| anyhow::anyhow!("set accept: {e:?}"))?
        .body(body)
        .reconnect(ReconnectOptions::reconnect(false).build())
        .build_with_transport(transport);

    let mut stream = Box::pin(client.stream());
    loop {
        match stream.try_next().await {
            Ok(Some(SSE::Event(ev))) => match serde_json::from_str::<StreamFrame>(&ev.data) {
                Ok(frame) => {
                    if tx.send(ClientEvent::Frame(frame)).await.is_err() {
                        return Ok(());
                    }
                }
                Err(e) => {
                    let _ = tx.send(ClientEvent::Error(format!("decode frame: {e}"))).await;
                    return Err(e).context("decode StreamFrame");
                }
            },
            Ok(Some(_)) => continue,
            Ok(None) => {
                let _ = tx.send(ClientEvent::Done).await;
                return Ok(());
            }
            Err(e) => {
                let _ = tx.send(ClientEvent::Error(format!("{e}"))).await;
                return Err(anyhow::anyhow!("{e}")).context("sse stream");
            }
        }
    }
}
