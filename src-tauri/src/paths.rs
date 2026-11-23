//! Where things live on disk.
//!
//! * `%APPDATA%\NoteTranscribe\settings.json`      – persisted settings
//! * `%APPDATA%\NoteTranscribe\transcripts\`       – transcripts (default)
//! * `%APPDATA%\NoteTranscribe\logs\app.log`       – log file (never contains secrets)
//! * `.env`, `prompts/summary_prompt.md`, `models/` – searched for next to the exe,
//!   in its parent folders (so `npm run tauri dev` finds the repo root), in the
//!   current directory, the Tauri resource dir and the data dir.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use meeting_core::settings::{Settings, SETTINGS_FILE_NAME};

pub const APP_DIR_NAME: &str = "NoteTranscribe";

static RESOURCE_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Record Tauri's resource dir (called once at startup).
pub fn set_resource_dir(p: PathBuf) {
    let _ = RESOURCE_DIR.set(p);
}

/// `%APPDATA%\NoteTranscribe` (falls back to the exe folder if APPDATA is unavailable).
pub fn data_dir() -> PathBuf {
    dirs::config_dir()
        .map(|d| d.join(APP_DIR_NAME))
        .or_else(|| exe_dir().map(|d| d.join("data")))
        .unwrap_or_else(|| PathBuf::from(APP_DIR_NAME))
}

pub fn settings_path() -> PathBuf {
    data_dir().join(SETTINGS_FILE_NAME)
}

pub fn logs_dir() -> PathBuf {
    data_dir().join("logs")
}

fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf))
}

/// Ordered list of base directories to search for project files.
fn search_bases() -> Vec<PathBuf> {
    let mut bases = Vec::new();
    let mut push_with_ancestors = |start: Option<PathBuf>, depth: usize| {
        if let Some(mut d) = start {
            for _ in 0..=depth {
                if !bases.contains(&d) {
                    bases.push(d.clone());
                }
                match d.parent() {
                    Some(p) => d = p.to_path_buf(),
                    None => break,
                }
            }
        }
    };
    push_with_ancestors(exe_dir(), 4);
    push_with_ancestors(std::env::current_dir().ok(), 3);
    if let Some(r) = RESOURCE_DIR.get() {
        if !bases.contains(r) {
            bases.push(r.clone());
        }
    }
    let dd = data_dir();
    if !bases.contains(&dd) {
        bases.push(dd);
    }
    bases
}

/// First existing `<base>/<rel>` across [`search_bases`].
pub fn find_project_path(rel: &str) -> Option<PathBuf> {
    search_bases().into_iter().map(|b| b.join(rel)).find(|p| p.exists())
}

/// Path of the editable summary prompt.
pub fn prompt_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SUMMARY_PROMPT_PATH") {
        let p = PathBuf::from(p.trim());
        if p.is_file() {
            return Some(p);
        }
    }
    find_project_path("prompts/summary_prompt.md").filter(|p| p.is_file())
}

/// Folder holding `ggml-*.bin`, `silero_vad.onnx` and the speaker-embedding model.
pub fn models_dir(settings: &Settings) -> PathBuf {
    let custom = settings.models_dir.trim();
    if !custom.is_empty() {
        return PathBuf::from(custom);
    }
    let bases = search_bases();
    // Prefer a models folder that actually contains a Whisper model.
    for b in &bases {
        let m = b.join("models");
        if m.is_dir() && !list_whisper_models(&m).is_empty() {
            return m;
        }
    }
    for b in &bases {
        let m = b.join("models");
        if m.is_dir() {
            return m;
        }
    }
    data_dir().join("models")
}

pub fn transcripts_dir(settings: &Settings) -> PathBuf {
    let custom = settings.transcripts_dir.trim();
    if custom.is_empty() {
        data_dir().join("transcripts")
    } else {
        PathBuf::from(custom)
    }
}

/// File name for a Whisper model name (`large-v3-turbo` -> `ggml-large-v3-turbo.bin`).
pub fn whisper_file_name(model: &str) -> String {
    let m = model.trim();
    if m.ends_with(".bin") {
        m.to_string()
    } else {
        format!("ggml-{m}.bin")
    }
}

/// Model names (without `ggml-` / `.bin`) present in a folder.
pub fn list_whisper_models(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            name.strip_prefix("ggml-")
                .and_then(|n| n.strip_suffix(".bin"))
                .filter(|n| !n.starts_with("silero")) // whisper.cpp's own VAD model, not ASR
                .map(str::to_string)
        })
        .collect();
    v.sort();
    v
}

pub const VAD_MODEL_FILE: &str = "silero_vad.onnx";

/// Speaker-embedding model: the first `*.onnx` in the models folder whose name
/// looks like a speaker model, or `SPEAKER_MODEL` from `.env`.
pub fn speaker_model_path(models: &Path) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SPEAKER_MODEL") {
        let p = PathBuf::from(p.trim());
        let p = if p.is_relative() { models.join(p) } else { p };
        if p.is_file() {
            return Some(p);
        }
    }
    let preferred = [
        "wespeaker_en_voxceleb_resnet34_LM.onnx",
        "wespeaker_en_voxceleb_resnet34.onnx",
        "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        "nemo_en_titanet_small.onnx",
    ];
    for name in preferred {
        let p = models.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    std::fs::read_dir(models).ok()?.flatten().map(|e| e.path()).find(|p| {
        let n = p.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
        n.ends_with(".onnx") && (n.contains("speaker") || n.contains("wespeaker") || n.contains("3dspeaker") || n.contains("titanet") || n.contains("ecapa"))
    })
}
