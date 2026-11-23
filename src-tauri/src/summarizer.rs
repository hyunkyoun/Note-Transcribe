//! Meeting summary via the Anthropic Messages API (raw HTTPS; there is no
//! official Rust SDK). Streams the response, retries with backoff, and
//! chunks very long transcripts.

use std::fmt;
use std::time::Duration;

use futures_util::StreamExt;
use meeting_core::prompt::{
    chunk_notes_prompt, chunk_transcript, combined_notes_transcript, estimate_tokens, render_template, standard_vars,
};
use meeting_core::sse::{interpret_event, SseParser, StreamItem};
use rand::Rng;
use serde_json::{json, Value};

use crate::config::ApiConfig;
use crate::events::Events;
use crate::meeting::Meeting;

const API_VERSION: &str = "2026-10-21";
const MAX_ATTEMPTS: u32 = 5;
/// Models that accept server-side refusal fallbacks (`fallbacks: "default"`).
const FALLBACK_MODELS: &[&str] = &["claude-opus-5-5", "claude-opus-5", "claude-fable-5-2", "claude-fable-5-1", "claude-sonnet-5-5"];

#[derive(Debug)]
pub struct SummaryError {
    pub message: String,
    pub retryable: bool,
    pub retry_after: Option<Duration>,
    pub too_long: bool,
}

impl SummaryError {
    fn fatal(message: impl Into<String>) -> Self {
        Self { message: message.into(), retryable: false, retry_after: None, too_long: false }
    }
    fn transient(message: impl Into<String>) -> Self {
        Self { message: message.into(), retryable: true, retry_after: None, too_long: false }
    }
}

impl fmt::Display for SummaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

fn utf8_take(buf: &mut Vec<u8>) -> String {
    match std::str::from_utf8(buf) {
        Ok(s) => {
            let s = s.to_string();
            buf.clear();
            s
        }
        Err(e) => {
            let valid = e.valid_up_to();
            let s = String::from_utf8_lossy(&buf[..valid]).to_string();
            buf.drain(..valid);
            s
        }
    }
}

async fn call_once(http: &reqwest::Client, cfg: &ApiConfig, prompt: &str) -> Result<String, SummaryError> {
    let key = cfg.anthropic_api_key.as_deref().ok_or_else(|| SummaryError::fatal("ANTHROPIC_API_KEY is not set in .env"))?;
    let mut body = json!({
        "model": cfg.claude_model,
        "max_tokens": cfg.claude_max_tokens,
        "stream": true,
        "messages": [{ "role": "user", "content": prompt }],
    });
    if let Some(effort) = &cfg.claude_effort {
        body["output_config"] = json!({ "effort": effort });
    }
    let mut req = http
        .post(format!("{}/v1/messages", cfg.anthropic_base_url.trim_end_matches('/')))
        .header("x-api-key", key)
        .header("anthropic-version", API_VERSION)
        .header("content-type", "application/json");
    if cfg.claude_fallbacks && FALLBACK_MODELS.contains(&cfg.claude_model.as_str()) {
        // If the model declines, the API re-runs the request on a fallback model.
        req = req.header("anthropic-beta", "server-side-fallback-2026-07-01");
        body["fallbacks"] = json!("default");
    }

    let resp = req.json(&body).send().await.map_err(|e| SummaryError::transient(format!("network error: {e}")))?;
    let status = resp.status();
    if !status.is_success() {
        let retry_after = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<f64>().ok())
            .map(Duration::from_secs_f64);
        let text = resp.text().await.unwrap_or_default();
        let detail = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| v.pointer("/error/message").and_then(Value::as_str).map(str::to_string))
            .unwrap_or_else(|| text.chars().take(300).collect());
        let code = status.as_u16();
        let retryable = matches!(code, 408 | 409 | 429 | 500..=599);
        let hint = match code {
            401 => " (check ANTHROPIC_API_KEY in .env)",
            404 => " (check CLAUDE_MODEL in .env)",
            _ => "",
        };
        return Err(SummaryError {
            too_long: code == 400 && detail.to_lowercase().contains("too long"),
            message: format!("Claude API returned {code}: {detail}{hint}"),
            retryable,
            retry_after,
        });
    }

    let mut stream = resp.bytes_stream();
    let mut parser = SseParser::default();
    let mut pending = Vec::<u8>::new();
    let mut out = String::new();
    let mut stop: Option<String> = None;
    'outer: while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| SummaryError::transient(format!("stream interrupted: {e}")))?;
        pending.extend_from_slice(&chunk);
        let text = utf8_take(&mut pending);
        for ev in parser.push(&text) {
            match interpret_event(&ev) {
                StreamItem::Text(t) => out.push_str(&t),
                StreamItem::Stop(s) => stop = Some(s),
                StreamItem::Model(m) if m != cfg.claude_model => log::info!("response served by fallback model {m}"),
                StreamItem::Error { kind, message } => {
                    let retryable = matches!(kind.as_str(), "overloaded_error" | "api_error" | "rate_limit_error");
                    return Err(SummaryError { message: format!("Claude stream error ({kind}): {message}"), retryable, retry_after: None, too_long: false });
                }
                StreamItem::Done => break 'outer,
                _ => {}
            }
        }
    }
    match stop.as_deref() {
        Some("refusal") => Err(SummaryError::fatal("Claude declined to summarize this transcript (stop_reason: refusal)")),
        Some("max_tokens") => {
            out.push_str("\n\n_(Summary was cut off at the max_tokens limit; raise CLAUDE_MAX_TOKENS in .env.)_");
            Ok(out)
        }
        None if out.trim().is_empty() => Err(SummaryError::transient("response stream ended before any text")),
        _ => Ok(out.trim().to_string()),
    }
}

/// One logical request with retries and exponential backoff.
async fn call_claude(http: &reqwest::Client, cfg: &ApiConfig, prompt: &str, events: &Events, label: &str) -> Result<String, SummaryError> {
    let mut attempt = 0;
    loop {
        attempt += 1;
        events.progress("summarizing", format!("{label} (attempt {attempt}/{MAX_ATTEMPTS})…"), None);
        match call_once(http, cfg, prompt).await {
            Ok(t) => return Ok(t),
            Err(e) if e.retryable && attempt < MAX_ATTEMPTS => {
                let backoff = Duration::from_secs(2u64.pow(attempt)) + Duration::from_millis(rand::rng().random_range(0..1000));
                let wait = e.retry_after.unwrap_or(backoff).min(Duration::from_secs(90));
                log::warn!("Claude call failed (attempt {attempt}): {e}; retrying in {:?}", wait);
                events.progress("retrying", format!("{e} — retrying in {}s", wait.as_secs()), None);
                tokio::time::sleep(wait).await;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Summarize a meeting using the user's prompt template.
pub async fn summarize(http: &reqwest::Client, cfg: &ApiConfig, template: &str, meeting: &Meeting, events: &Events) -> Result<String, SummaryError> {
    let transcript = meeting.transcript_text();
    if transcript.trim().is_empty() {
        return Err(SummaryError::fatal("the transcript is empty; nothing to summarize"));
    }
    let speakers = meeting.display_speakers();
    let vars = |t: &str| standard_vars(t, &meeting.started_label, &meeting.duration_label(), &speakers);

    let full = render_template(template, &vars(&transcript));
    let mut budget = cfg.claude_max_input_tokens.max(10_000);
    if estimate_tokens(&full) <= budget {
        match call_claude(http, cfg, &full, events, "Summarizing with Claude").await {
            Ok(s) => return Ok(s),
            Err(e) if e.too_long => {
                budget = estimate_tokens(&full) / 2;
                log::warn!("prompt too long for model; switching to chunked summary");
            }
            Err(e) => return Err(e),
        }
    }

    // Long meeting: notes per chunk, then the user's template over the notes.
    let max_chars = ((budget as f64) * 3.5 * 0.9) as usize;
    let chunks = chunk_transcript(&transcript, max_chars.max(10_000));
    let n = chunks.len();
    let mut notes = Vec::with_capacity(n);
    for (i, chunk) in chunks.iter().enumerate() {
        let label = format!("Long meeting: summarizing part {}/{n}", i + 1);
        notes.push(call_claude(http, cfg, &chunk_notes_prompt(i, n, chunk), events, &label).await?);
    }
    let combined = render_template(template, &vars(&combined_notes_transcript(&notes)));
    call_claude(http, cfg, &combined, events, "Combining part summaries").await
}
