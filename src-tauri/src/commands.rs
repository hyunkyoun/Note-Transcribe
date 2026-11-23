//! Tauri commands invoked from the frontend.

use std::sync::atomic::Ordering;

use meeting_core::labels::{should_diarize, Renames};
use meeting_core::settings::Settings;
use meeting_core::Source;
use serde::Serialize;
use tauri::{AppHandle, Manager, State};
use tauri_plugin_opener::OpenerExt;

use crate::audio::{self, DeviceList};
use crate::config::{self, ApiConfig, ConfigStatus};
use crate::diarize;
use crate::discord;
use crate::events::Events;
use crate::meeting::{Meeting, MeetingView};
use crate::paths;
use crate::session::ActiveRecording;
use crate::state::AppState;
use crate::summarizer;
use crate::transcriber::{self, EngineInfo};

/// Built-in copy of the default prompt, used only if prompts/summary_prompt.md can't be found.
const DEFAULT_PROMPT: &str = include_str!("../../prompts/summary_prompt.md");

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppStatus {
    pub recording: bool,
    pub session_id: Option<String>,
    pub elapsed_ms: Option<u64>,
    pub pending_jobs: usize,
    pub busy: bool,
    pub config: ConfigStatus,
    pub prompt_path: Option<String>,
    pub models_dir: String,
    pub whisper_models: Vec<String>,
    pub has_vad_model: bool,
    pub speaker_model: Option<String>,
    pub data_dir: String,
    pub settings_path: String,
    pub transcripts_dir: String,
    pub cuda_build: bool,
    pub engine: Option<EngineInfo>,
    pub settings_warning: Option<String>,
}

#[tauri::command]
pub fn get_settings(state: State<'_, AppState>) -> Settings {
    state.settings.get()
}

#[tauri::command]
pub fn update_settings(state: State<'_, AppState>, settings: Settings) -> Settings {
    state.settings.update(settings)
}

#[tauri::command]
pub async fn list_devices() -> Result<DeviceList, String> {
    tauri::async_runtime::spawn_blocking(audio::list_devices).await.map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_status(state: State<'_, AppState>) -> AppStatus {
    let s = state.settings.get();
    let models = paths::models_dir(&s);
    let rec = state.recording.lock();
    let engine = rec
        .as_ref()
        .and_then(|r| r.shared_engine_info())
        .or_else(|| state.last_engine.lock().clone());
    AppStatus {
        recording: rec.is_some(),
        session_id: rec.as_ref().map(|r| r.shared.id.clone()),
        elapsed_ms: rec.as_ref().map(|r| r.elapsed_ms()),
        pending_jobs: rec.as_ref().map(|r| r.pending_jobs()).unwrap_or(0),
        busy: state.busy.load(Ordering::SeqCst),
        config: ApiConfig::from_env().status(),
        prompt_path: paths::prompt_path().map(|p| p.display().to_string()),
        whisper_models: paths::list_whisper_models(&models),
        has_vad_model: models.join(paths::VAD_MODEL_FILE).is_file(),
        speaker_model: paths::speaker_model_path(&models).map(|p| p.file_name().unwrap_or_default().to_string_lossy().to_string()),
        models_dir: models.display().to_string(),
        data_dir: paths::data_dir().display().to_string(),
        settings_path: state.settings.path().display().to_string(),
        transcripts_dir: paths::transcripts_dir(&s).display().to_string(),
        cuda_build: cfg!(feature = "cuda"),
        engine,
        settings_warning: state.settings.load_warning.clone(),
    }
}

#[tauri::command]
pub fn reload_env() -> ConfigStatus {
    config::load_env();
    ApiConfig::from_env().status()
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StartInfo {
    pub session_id: String,
    pub transcripts_dir: String,
}

#[tauri::command]
pub async fn start_recording(app: AppHandle) -> Result<StartInfo, String> {
    let state = app.state::<AppState>();
    if state.recording.lock().is_some() {
        return Err("Already recording".into());
    }
    let settings = state.settings.get();
    let models_dir = paths::models_dir(&settings);
    let transcripts_dir = paths::transcripts_dir(&settings);
    // Fast preflight so a missing model is reported immediately.
    transcriber::resolve_model(&models_dir, &settings).map_err(|e| format!("{e:#}"))?;

    let events = Events(app.clone());
    let needs_diarization = (settings.capture_desktop && should_diarize(Source::Desktop, settings.mic_is_only_me))
        || (settings.capture_mic && should_diarize(Source::Mic, settings.mic_is_only_me));
    let extractor = if needs_diarization {
        let cached = state.extractor.lock().clone();
        match cached {
            Some(e) => Some(e),
            None => {
                let dir = models_dir.clone();
                match tauri::async_runtime::spawn_blocking(move || diarize::load_extractor(&dir)).await {
                    Ok(Ok(e)) => {
                        *state.extractor.lock() = Some(e.clone());
                        Some(e)
                    }
                    Ok(Err(msg)) => {
                        events.status(None, audio::Level::Warn, msg);
                        None
                    }
                    Err(e) => {
                        events.status(None, audio::Level::Warn, format!("speaker model: {e}"));
                        None
                    }
                }
            }
        }
    } else {
        None
    };

    let gpu_expected = settings.use_gpu && cfg!(feature = "cuda");
    let rec = ActiveRecording::start(
        settings,
        models_dir,
        transcripts_dir.clone(),
        state.engine.clone(),
        extractor,
        gpu_expected,
        events.clone(),
    )
    .map_err(|e| format!("{e:#}"))?;
    let info = StartInfo { session_id: rec.shared.id.clone(), transcripts_dir: transcripts_dir.display().to_string() };
    *state.recording.lock() = Some(rec);
    events.recording_state("recording");
    Ok(info)
}

#[tauri::command]
pub async fn stop_recording(app: AppHandle) -> Result<MeetingView, String> {
    let state = app.state::<AppState>();
    let rec = state.recording.lock().take().ok_or("Not recording")?;
    let events = Events(app.clone());
    events.recording_state("finalizing");
    let engine_info = rec.shared_engine_info();
    let meeting = tauri::async_runtime::spawn_blocking(move || rec.stop(false)).await.map_err(|e| e.to_string())?;

    if engine_info.is_some() {
        *state.last_engine.lock() = engine_info;
    }
    let settings = state.settings.get();
    if !settings.keep_model_loaded {
        *state.extractor.lock() = None;
    }
    let view = meeting.view();
    *state.meeting.lock() = Some(meeting);
    events.recording_state("idle");

    if settings.auto_summarize_on_stop {
        let app2 = app.clone();
        tauri::async_runtime::spawn(async move {
            if let Err(e) = summarize_and_send(app2.clone(), None).await {
                Events(app2).progress("error", e, Some(false));
            }
        });
    }
    Ok(view)
}

#[tauri::command]
pub fn get_meeting(state: State<'_, AppState>) -> Option<MeetingView> {
    state.meeting.lock().as_ref().map(Meeting::view)
}

#[tauri::command]
pub fn get_transcript(state: State<'_, AppState>) -> Option<String> {
    state.meeting.lock().as_ref().map(Meeting::transcript_text)
}

fn apply_renames(state: &AppState, renames: Renames) -> Result<MeetingView, String> {
    let mut guard = state.meeting.lock();
    let m = guard.as_mut().ok_or("No finished meeting yet")?;
    m.renames = renames;
    m.save().map_err(|e| format!("{e:#}"))?;
    Ok(m.view())
}

#[tauri::command]
pub fn set_renames(state: State<'_, AppState>, renames: Renames) -> Result<MeetingView, String> {
    apply_renames(&state, renames)
}

/// Clears the busy flag when dropped.
struct BusyGuard<'a>(&'a std::sync::atomic::AtomicBool);
impl Drop for BusyGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

fn load_prompt(events: &Events) -> String {
    match paths::prompt_path().and_then(|p| std::fs::read_to_string(p).ok()) {
        Some(t) if !t.trim().is_empty() => t,
        _ => {
            events.status(None, audio::Level::Warn, "prompts/summary_prompt.md not found; using the built-in default prompt");
            DEFAULT_PROMPT.to_string()
        }
    }
}

/// Apply renames, summarize with Claude, then (if enabled) post to Discord.
#[tauri::command]
pub async fn summarize_and_send(app: AppHandle, renames: Option<Renames>) -> Result<MeetingView, String> {
    let state = app.state::<AppState>();
    if state.busy.swap(true, Ordering::SeqCst) {
        return Err("A summary is already in progress".into());
    }
    let _guard = BusyGuard(&state.busy);
    let events = Events(app.clone());
    if let Some(r) = renames {
        apply_renames(&state, r)?;
    }
    let meeting = state.meeting.lock().clone().ok_or("No finished meeting yet")?;
    let transcript_path = meeting.txt_path().display().to_string();

    config::load_env(); // pick up .env edits without restarting
    let cfg = ApiConfig::from_env();
    if cfg.anthropic_api_key.is_none() {
        let msg = format!("ANTHROPIC_API_KEY is not set in .env. Your transcript is saved at {transcript_path}");
        events.progress("error", msg.clone(), Some(false));
        return Err(msg);
    }
    let template = load_prompt(&events);
    let summary = match summarizer::summarize(&state.http, &cfg, &template, &meeting, &events).await {
        Ok(s) => s,
        Err(e) => {
            let msg = format!("Summary failed: {e}. Nothing is lost: the transcript is saved at {transcript_path}");
            log::error!("{msg}");
            events.progress("error", msg.clone(), Some(false));
            return Err(msg);
        }
    };

    let view = {
        let mut guard = state.meeting.lock();
        let m = guard.as_mut().ok_or("Meeting disappeared")?;
        m.summary = Some(summary);
        m.save().map_err(|e| format!("{e:#}"))?;
        m.view()
    };
    events.progress("summary", format!("Summary saved to {}", view.summary_path.clone().unwrap_or_default()), Some(true));

    if state.settings.get().post_to_discord {
        return post_to_discord(&app, true).await;
    }
    Ok(view)
}

async fn post_to_discord(app: &AppHandle, include_summary: bool) -> Result<MeetingView, String> {
    let state = app.state::<AppState>();
    let events = Events(app.clone());
    config::load_env();
    let cfg = ApiConfig::from_env();
    let Some(url) = cfg.discord_webhook_url else {
        let msg = "DISCORD_WEBHOOK_URL is not set in .env; skipped Discord".to_string();
        events.progress("discord", msg.clone(), Some(false));
        return Err(msg);
    };
    let meeting = {
        let g = state.meeting.lock();
        g.clone()
    };
    let meeting = match meeting.or_else(Meeting::load_last) {
        Some(m) => m,
        None => return Err("No meeting to send yet".into()),
    };
    events.progress("discord", "Posting to Discord…", None);
    match discord::post_meeting(&state.http, &url, &meeting, include_summary).await {
        Ok(n) => {
            events.progress("discord", format!("Posted to Discord ({n} messages)"), Some(true));
            let mut g = state.meeting.lock();
            let m = g.get_or_insert(meeting);
            m.discord_posted = true;
            let _ = m.save();
            Ok(m.view())
        }
        Err(e) => {
            log::error!("Discord post failed: {e}");
            events.progress("discord", format!("Discord post failed: {e}"), Some(false));
            Err(e)
        }
    }
}

/// "Re-send last summary to Discord" (or transcript only).
#[tauri::command]
pub async fn send_to_discord(app: AppHandle, transcript_only: bool) -> Result<MeetingView, String> {
    let state = app.state::<AppState>();
    if state.busy.swap(true, Ordering::SeqCst) {
        return Err("Another summary/Discord job is in progress".into());
    }
    let _guard = BusyGuard(&state.busy);
    post_to_discord(&app, !transcript_only).await
}

#[tauri::command]
pub fn open_location(app: AppHandle, which: String) -> Result<(), String> {
    let state = app.state::<AppState>();
    let s = state.settings.get();
    let path = match which.as_str() {
        "prompt" => paths::prompt_path().ok_or("prompts/summary_prompt.md not found")?,
        "transcripts" => paths::transcripts_dir(&s),
        "models" => paths::models_dir(&s),
        "data" => paths::data_dir(),
        "env" => config::loaded_env_path().ok_or(".env not found; copy .env.example to .env first")?,
        "transcript" => state.meeting.lock().as_ref().map(|m| m.txt_path()).ok_or("no transcript yet")?,
        other => return Err(format!("unknown location {other}")),
    };
    if matches!(which.as_str(), "transcripts" | "models" | "data") {
        std::fs::create_dir_all(&path).ok();
    }
    app.opener().open_path(path.to_string_lossy(), None::<&str>).map_err(|e| e.to_string())
}
