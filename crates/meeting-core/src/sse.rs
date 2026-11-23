//! Minimal Server-Sent Events parser plus interpretation of the Anthropic
//! Messages API streaming events. Kept here (pure, no I/O) so it is testable.

use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

/// Incremental SSE parser. Push raw text as it arrives; complete events come out.
#[derive(Debug, Default)]
pub struct SseParser {
    buf: String,
}

impl SseParser {
    pub fn push(&mut self, chunk: &str) -> Vec<SseEvent> {
        self.buf.push_str(&chunk.replace("\r\n", "\n").replace('\r', "\n"));
        let mut events = Vec::new();
        while let Some(pos) = self.buf.find("\n\n") {
            let block: String = self.buf.drain(..pos + 2).collect();
            let mut event = None;
            let mut data: Vec<&str> = Vec::new();
            for line in block.lines() {
                if line.is_empty() || line.starts_with(':') {
                    continue;
                }
                let (field, value) = match line.split_once(':') {
                    Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
                    None => (line, ""),
                };
                match field {
                    "event" => event = Some(value.to_string()),
                    "data" => data.push(value),
                    _ => {}
                }
            }
            if event.is_some() || !data.is_empty() {
                events.push(SseEvent { event, data: data.join("\n") });
            }
        }
        events
    }
}

/// What a Messages API stream event means for us.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamItem {
    /// Visible answer text.
    Text(String),
    /// Final stop reason (`end_turn`, `max_tokens`, `refusal`, ...).
    Stop(String),
    /// The model that actually served the response (may differ if a server-side fallback ran).
    Model(String),
    /// Error event inside the stream (e.g. `overloaded_error`).
    Error { kind: String, message: String },
    /// Message finished.
    Done,
    /// Anything we don't need (thinking deltas, pings, block start/stop...).
    Ignore,
}

/// Interpret one SSE event's JSON payload.
pub fn interpret_event(ev: &SseEvent) -> StreamItem {
    let Ok(v) = serde_json::from_str::<Value>(&ev.data) else {
        return StreamItem::Ignore;
    };
    let ty = v.get("type").and_then(Value::as_str).unwrap_or_default();
    match ty {
        "message_start" => v
            .pointer("/message/model")
            .and_then(Value::as_str)
            .map(|m| StreamItem::Model(m.to_string()))
            .unwrap_or(StreamItem::Ignore),
        "content_block_delta" => match v.pointer("/delta/type").and_then(Value::as_str) {
            Some("text_delta") => StreamItem::Text(
                v.pointer("/delta/text").and_then(Value::as_str).unwrap_or_default().to_string(),
            ),
            _ => StreamItem::Ignore,
        },
        "message_delta" => v
            .pointer("/delta/stop_reason")
            .and_then(Value::as_str)
            .map(|s| StreamItem::Stop(s.to_string()))
            .unwrap_or(StreamItem::Ignore),
        "message_stop" => StreamItem::Done,
        "error" => StreamItem::Error {
            kind: v.pointer("/error/type").and_then(Value::as_str).unwrap_or("error").to_string(),
            message: v.pointer("/error/message").and_then(Value::as_str).unwrap_or_default().to_string(),
        },
        _ => StreamItem::Ignore,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_events_across_chunk_boundaries() {
        let mut p = SseParser::default();
        let mut evs = p.push("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",");
        assert!(evs.is_empty());
        evs.extend(p.push("\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\nevent: ping\r\ndata: {\"type\":\"ping\"}\r\n\r\n"));
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].event.as_deref(), Some("content_block_delta"));
        assert_eq!(interpret_event(&evs[0]), StreamItem::Text("Hi".into()));
        assert_eq!(interpret_event(&evs[1]), StreamItem::Ignore);
    }

    #[test]
    fn interprets_lifecycle() {
        let ev = |d: &str| SseEvent { event: None, data: d.into() };
        assert_eq!(
            interpret_event(&ev(r#"{"type":"message_start","message":{"model":"claude-opus-5-5"}}"#)),
            StreamItem::Model("claude-opus-5-5".into())
        );
        assert_eq!(
            interpret_event(&ev(r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{}}"#)),
            StreamItem::Stop("end_turn".into())
        );
        assert_eq!(interpret_event(&ev(r#"{"type":"message_stop"}"#)), StreamItem::Done);
        assert_eq!(
            interpret_event(&ev(r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#)),
            StreamItem::Error { kind: "overloaded_error".into(), message: "Overloaded".into() }
        );
        // Thinking deltas are not part of the visible summary.
        assert_eq!(
            interpret_event(&ev(r#"{"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"x"}}"#)),
            StreamItem::Ignore
        );
        assert_eq!(interpret_event(&ev("not json")), StreamItem::Ignore);
    }

    #[test]
    fn ignores_comments_and_multiline_data() {
        let mut p = SseParser::default();
        let evs = p.push(": keepalive\n\ndata: line1\ndata: line2\n\n");
        assert_eq!(evs, vec![SseEvent { event: None, data: "line1\nline2".into() }]);
    }
}
