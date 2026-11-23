//! Typed events sent to the frontend.

use meeting_core::Source;
use serde::Serialize;
use tauri::{AppHandle, Emitter};

use crate::audio::Level;
use crate::transcriber::EngineInfo;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptionEvent {
    /// "partial" (may change), "final", or "clear" (drop the partial line for `source`).
    pub kind: &'static str,
    pub source: Source,
    pub id: Option<u64>,
    pub start_ms: u64,
    pub end_ms: u64,
    pub speaker: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StatusEvent {
    pub source: Option<Source>,
    pub level: Level,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineEvent {
    pub state: &'static str,
    pub message: String,
    pub info: Option<EngineInfo>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProgressEvent {
    pub stage: &'static str,
    pub message: String,
    pub ok: Option<bool>,
}

#[derive(Clone)]
pub struct Events(pub AppHandle);

impl Events {
    pub fn caption(&self, e: CaptionEvent) {
        let _ = self.0.emit("caption", e);
    }
    pub fn status(&self, source: Option<Source>, level: Level, message: impl Into<String>) {
        let _ = self.0.emit("status", StatusEvent { source, level, message: message.into() });
    }
    pub fn engine(&self, state: &'static str, message: impl Into<String>, info: Option<EngineInfo>) {
        let _ = self.0.emit("engine", EngineEvent { state, message: message.into(), info });
    }
    pub fn progress(&self, stage: &'static str, message: impl Into<String>, ok: Option<bool>) {
        let _ = self.0.emit("progress", ProgressEvent { stage, message: message.into(), ok });
    }
    pub fn recording_state(&self, state: &'static str) {
        let _ = self.0.emit("recording-state", state);
    }
}
