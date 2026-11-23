//! Posting the summary and transcript to a Discord channel via webhook.

use std::time::Duration;

use meeting_core::discord_split::{split_message, transcript_fits_inline, DISCORD_LIMIT};
use reqwest::multipart::{Form, Part};
use serde_json::{json, Value};

use crate::meeting::Meeting;

const MAX_FILE_BYTES: usize = 9_500_000;

fn validate(url: &str) -> Result<(), String> {
    let ok = url.starts_with("https://")
        && url.contains("/api/webhooks/")
        && (url.contains("discord.com") || url.contains("discordapp.com"));
    if ok {
        Ok(())
    } else {
        Err("DISCORD_WEBHOOK_URL doesn't look like a Discord webhook URL (https://discord.com/api/webhooks/…)".into())
    }
}

fn wait_url(url: &str) -> String {
    if url.contains('?') {
        format!("{url}&wait=true")
    } else {
        format!("{url}?wait=true")
    }
}

/// POST with Discord rate-limit handling. `make` builds a fresh request each attempt.
async fn send<F>(make: F) -> Result<(), String>
where
    F: Fn() -> reqwest::RequestBuilder,
{
    for _ in 0..6 {
        let resp = make().send().await.map_err(|e| format!("network error posting to Discord: {e}"))?;
        let status = resp.status();
        if status.is_success() {
            return Ok(());
        }
        let text = resp.text().await.unwrap_or_default();
        if status.as_u16() == 429 {
            let secs = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| v.get("retry_after").and_then(Value::as_f64))
                .unwrap_or(1.0);
            tokio::time::sleep(Duration::from_secs_f64(secs.clamp(0.2, 30.0))).await;
            continue;
        }
        if status.is_server_error() {
            tokio::time::sleep(Duration::from_secs(2)).await;
            continue;
        }
        let detail: String = text.chars().take(300).collect();
        return Err(format!("Discord returned {}: {detail}", status.as_u16()));
    }
    Err("Discord kept rate-limiting or failing; gave up".into())
}

async fn post_text(http: &reqwest::Client, url: &str, content: &str) -> Result<(), String> {
    // allowed_mentions: none, so "@everyone" in a transcript can't ping the channel.
    let body = json!({ "content": content, "allowed_mentions": { "parse": [] } });
    send(|| http.post(wait_url(url)).json(&body)).await
}

async fn post_file(http: &reqwest::Client, url: &str, content: &str, filename: &str, bytes: Vec<u8>) -> Result<(), String> {
    if bytes.len() > MAX_FILE_BYTES {
        return Err(format!("transcript is {:.1} MB, over Discord's upload limit", bytes.len() as f64 / 1e6));
    }
    let payload = json!({
        "content": content,
        "allowed_mentions": { "parse": [] },
        "attachments": [{ "id": 0, "filename": filename }],
    })
    .to_string();
    send(|| {
        let part = Part::bytes(bytes.clone()).file_name(filename.to_string()).mime_str("text/plain; charset=utf-8").expect("valid mime");
        let form = Form::new().text("payload_json", payload.clone()).part("files[0]", part);
        http.post(wait_url(url)).multipart(form)
    })
    .await
}

/// Post the summary (split into ≤2000-char messages) and then the transcript
/// (inline if short, otherwise as a .txt attachment). Returns the number of messages sent.
pub async fn post_meeting(http: &reqwest::Client, webhook: &str, meeting: &Meeting, include_summary: bool) -> Result<usize, String> {
    validate(webhook)?;
    log::info!("posting meeting {} to Discord webhook {}", meeting.id, crate::logging::redact_url(webhook));
    let header = format!(
        "**Meeting notes · {} · {}**\nSpeakers: {}",
        meeting.started_label,
        meeting.duration_label(),
        meeting.display_speakers().join(", ")
    );
    let body = match (&meeting.summary, include_summary) {
        (Some(s), true) => s.clone(),
        _ => "_No AI summary for this meeting; the transcript follows._".to_string(),
    };
    let mut sent = 0;
    for chunk in split_message(&format!("{header}\n\n{body}"), DISCORD_LIMIT) {
        post_text(http, webhook, &chunk).await?;
        sent += 1;
        tokio::time::sleep(Duration::from_millis(350)).await;
    }

    let transcript = meeting.transcript_text();
    if transcript_fits_inline(&transcript, DISCORD_LIMIT) {
        let safe = transcript.replace("```", "'''");
        post_text(http, webhook, &format!("**Transcript**\n```\n{safe}```")).await?;
    } else {
        let name = format!("{}_transcript.txt", meeting.id);
        let file = format!("Meeting {} ({})\n\n{}", meeting.started_label, meeting.duration_label(), transcript);
        post_file(http, webhook, "**Full transcript** (attached)", &name, file.into_bytes()).await?;
    }
    Ok(sent + 1)
}
