//! User settings persisted to `settings.json` in the per-user data directory.
//!
//! Secrets (API keys, webhook URLs, tokens) are deliberately NOT part of this
//! struct: they live only in `.env`.
//!
//! Loading never fails: a missing file gives defaults, a corrupt file is moved
//! aside (so you can inspect it) and defaults are used, and a file with a few
//! bad fields keeps every field that is still valid.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const SETTINGS_FILE_NAME: &str = "settings.json";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Used as the mic speaker label when `mic_is_only_me` is on. Empty = "Me".
    pub my_name: String,
    /// On: every mic segment is labelled with `my_name`, no mic diarization.
    /// Off: diarize the mic too ("Mic Speaker 1", "Mic Speaker 2", ...).
    pub mic_is_only_me: bool,

    /// Capture WASAPI loopback of an output device (the other participants).
    pub capture_desktop: bool,
    /// Capture the microphone.
    pub capture_mic: bool,
    /// Output device name to loopback-capture. `None` = Windows default output.
    pub output_device: Option<String>,
    /// Microphone device name. `None` = Windows default input.
    pub mic_device: Option<String>,

    /// Whisper model name; file is `ggml-<name>.bin` in the models folder.
    pub whisper_model: String,
    /// Smaller model used if the main one is missing or when running on CPU.
    pub fallback_model: String,
    /// Try CUDA first. If the GPU can't be used we fall back to CPU automatically.
    pub use_gpu: bool,
    /// ISO-639-1 code like "en", or "auto" for per-segment auto-detect.
    pub language: String,
    /// Optional Whisper initial prompt (names, jargon) to improve recognition.
    pub initial_prompt: String,

    /// Cosine-similarity threshold for "same speaker" (0.1 - 0.95).
    pub diarization_threshold: f32,
    /// Upper bound on distinct speakers per stream.
    pub max_speakers: usize,
    /// Drop mic segments that are just the desktop audio leaking into the mic.
    pub suppress_mic_echo: bool,

    /// Save 16 kHz mono WAV files of each stream next to the transcript.
    pub save_raw_audio: bool,
    /// Live caption font size in px.
    pub caption_font_size: u32,

    /// Generate the summary immediately on Stop (skips the rename step).
    pub auto_summarize_on_stop: bool,
    /// Post summary + transcript to Discord after summarizing.
    pub post_to_discord: bool,
    /// Keep the Whisper model in (V)RAM between recordings (faster start, more memory).
    pub keep_model_loaded: bool,

    /// Override folder for model files. Empty = auto-detect.
    pub models_dir: String,
    /// Override folder for transcripts. Empty = `<data dir>/transcripts`.
    pub transcripts_dir: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            my_name: String::new(),
            mic_is_only_me: true,
            capture_desktop: true,
            capture_mic: true,
            output_device: None,
            mic_device: None,
            whisper_model: "large-v3-turbo".into(),
            fallback_model: "base".into(),
            use_gpu: true,
            language: "auto".into(),
            initial_prompt: String::new(),
            diarization_threshold: 0.5,
            max_speakers: 8,
            suppress_mic_echo: true,
            save_raw_audio: false,
            caption_font_size: 16,
            auto_summarize_on_stop: false,
            post_to_discord: true,
            keep_model_loaded: false,
            models_dir: String::new(),
            transcripts_dir: String::new(),
        }
    }
}

impl Settings {
    /// Clamp numeric fields into sane ranges and normalize strings.
    pub fn sanitized(mut self) -> Self {
        self.caption_font_size = self.caption_font_size.clamp(10, 48);
        if !self.diarization_threshold.is_finite() {
            self.diarization_threshold = Settings::default().diarization_threshold;
        }
        self.diarization_threshold = self.diarization_threshold.clamp(0.1, 0.95);
        self.max_speakers = self.max_speakers.clamp(1, 20);
        self.language = self.language.trim().to_ascii_lowercase();
        if self.language.is_empty() {
            self.language = "auto".into();
        }
        if self.whisper_model.trim().is_empty() {
            self.whisper_model = Settings::default().whisper_model;
        }
        if self.fallback_model.trim().is_empty() {
            self.fallback_model = Settings::default().fallback_model;
        }
        for dev in [&mut self.output_device, &mut self.mic_device] {
            if dev.as_deref().map(str::trim).is_some_and(str::is_empty) {
                *dev = None;
            }
        }
        self
    }
}

/// What happened when loading.
#[derive(Debug, Clone, PartialEq)]
pub enum LoadOutcome {
    /// File parsed cleanly.
    Loaded,
    /// File parsed but some fields were invalid and replaced by defaults.
    Partial { dropped_fields: Vec<String> },
    /// No file yet; defaults used.
    Missing,
    /// File unreadable / not JSON; defaults used. `backup` is where the bad file was moved.
    Corrupt { error: String, backup: Option<PathBuf> },
}

/// Load settings, never failing. See module docs.
pub fn load_settings(path: &Path) -> (Settings, LoadOutcome) {
    let raw = match fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return (Settings::default(), LoadOutcome::Missing),
        Err(e) => {
            return (
                Settings::default(),
                LoadOutcome::Corrupt { error: e.to_string(), backup: None },
            )
        }
    };
    let text = String::from_utf8_lossy(&raw);
    let value: Value = match serde_json::from_str(text.trim_start_matches('\u{feff}')) {
        Ok(v) => v,
        Err(e) => {
            let backup = backup_corrupt(path);
            return (Settings::default(), LoadOutcome::Corrupt { error: e.to_string(), backup });
        }
    };
    let Value::Object(file_obj) = value else {
        let backup = backup_corrupt(path);
        return (
            Settings::default(),
            LoadOutcome::Corrupt { error: "settings.json is not a JSON object".into(), backup },
        );
    };

    // Fast path: whole object valid.
    if let Ok(s) = serde_json::from_value::<Settings>(Value::Object(file_obj.clone())) {
        return (s.sanitized(), LoadOutcome::Loaded);
    }

    // Field-by-field recovery: keep each field that deserializes on its own.
    let mut merged = match serde_json::to_value(Settings::default()) {
        Ok(Value::Object(m)) => m,
        _ => unreachable!("Settings always serializes to an object"),
    };
    let mut dropped = Vec::new();
    for (k, v) in file_obj {
        if !merged.contains_key(&k) {
            continue; // unknown keys are ignored (forward/backward compat)
        }
        let mut trial = merged.clone();
        trial.insert(k.clone(), v);
        if serde_json::from_value::<Settings>(Value::Object(trial.clone())).is_ok() {
            merged = trial;
        } else {
            dropped.push(k);
        }
    }
    let s = serde_json::from_value::<Settings>(Value::Object(merged)).unwrap_or_default();
    (s.sanitized(), LoadOutcome::Partial { dropped_fields: dropped })
}

fn backup_corrupt(path: &Path) -> Option<PathBuf> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let backup = path.with_file_name(format!("settings.corrupt-{stamp}.json"));
    fs::rename(path, &backup).ok().map(|_| backup)
}

/// Save atomically (write a temp file next to the target, then rename) so a
/// crash mid-write can't leave a truncated settings.json.
pub fn save_settings(path: &Path, settings: &Settings) -> io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let json = serde_json::to_string_pretty(settings).map_err(io::Error::other)?;
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, json.as_bytes())?;
    // On Windows rename over an existing file works with std (MoveFileExW REPLACE_EXISTING).
    fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_gives_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let (s, o) = load_settings(&dir.path().join("settings.json"));
        assert_eq!(s, Settings::default());
        assert_eq!(o, LoadOutcome::Missing);
    }

    #[test]
    fn round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("nested").join("settings.json");
        let s = Settings {
            my_name: "Eun".into(),
            mic_is_only_me: false,
            output_device: Some("Headphones (Realtek)".into()),
            mic_device: Some("Mic (USB)".into()),
            whisper_model: "small".into(),
            language: "en".into(),
            save_raw_audio: true,
            caption_font_size: 22,
            ..Default::default()
        };
        save_settings(&p, &s).unwrap();
        let (loaded, o) = load_settings(&p);
        assert_eq!(o, LoadOutcome::Loaded);
        assert_eq!(loaded, s);
        // No temp file left behind.
        assert!(!p.with_extension("json.tmp").exists());
    }

    #[test]
    fn corrupt_file_gives_defaults_and_is_backed_up() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("settings.json");
        fs::write(&p, "{ this is not json").unwrap();
        let (s, o) = load_settings(&p);
        assert_eq!(s, Settings::default());
        match o {
            LoadOutcome::Corrupt { backup: Some(b), .. } => assert!(b.exists()),
            other => panic!("unexpected {other:?}"),
        }
        assert!(!p.exists(), "corrupt file should be moved aside");
    }

    #[test]
    fn non_object_json_is_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("settings.json");
        fs::write(&p, "[1,2,3]").unwrap();
        let (s, o) = load_settings(&p);
        assert_eq!(s, Settings::default());
        assert!(matches!(o, LoadOutcome::Corrupt { .. }));
    }

    #[test]
    fn empty_file_is_corrupt_not_a_crash() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("settings.json");
        fs::write(&p, "").unwrap();
        let (s, o) = load_settings(&p);
        assert_eq!(s, Settings::default());
        assert!(matches!(o, LoadOutcome::Corrupt { .. }));
    }

    #[test]
    fn bad_fields_are_dropped_good_fields_kept() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("settings.json");
        fs::write(
            &p,
            r#"{"my_name":"Eun","mic_is_only_me":"yes please","caption_font_size":20,"future_field":1}"#,
        )
        .unwrap();
        let (s, o) = load_settings(&p);
        assert_eq!(s.my_name, "Eun");
        assert_eq!(s.caption_font_size, 20);
        assert!(s.mic_is_only_me, "invalid value falls back to default");
        assert_eq!(o, LoadOutcome::Partial { dropped_fields: vec!["mic_is_only_me".into()] });
    }

    #[test]
    fn missing_fields_use_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("settings.json");
        fs::write(&p, r#"{"my_name":"Eun"}"#).unwrap();
        let (s, o) = load_settings(&p);
        assert_eq!(o, LoadOutcome::Loaded);
        assert_eq!(s.my_name, "Eun");
        assert_eq!(s.whisper_model, "large-v3-turbo");
        assert!(s.mic_is_only_me);
        assert!(!s.save_raw_audio);
    }

    #[test]
    fn utf8_bom_is_tolerated() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("settings.json");
        fs::write(&p, "\u{feff}{\"my_name\":\"Eun\"}").unwrap();
        let (s, o) = load_settings(&p);
        assert_eq!(o, LoadOutcome::Loaded);
        assert_eq!(s.my_name, "Eun");
    }

    #[test]
    fn sanitize_clamps_values() {
        let s = Settings {
            caption_font_size: 500,
            diarization_threshold: f32::NAN,
            max_speakers: 0,
            language: "  EN ".into(),
            output_device: Some("  ".into()),
            whisper_model: "".into(),
            ..Default::default()
        }
        .sanitized();
        assert_eq!(s.caption_font_size, 48);
        assert_eq!(s.diarization_threshold, 0.5);
        assert_eq!(s.max_speakers, 1);
        assert_eq!(s.language, "en");
        assert_eq!(s.output_device, None);
        assert_eq!(s.whisper_model, "large-v3-turbo");
    }

    #[test]
    fn secrets_are_not_settings() {
        let json = serde_json::to_string(&Settings::default()).unwrap().to_lowercase();
        for forbidden in ["api_key", "webhook", "token", "anthropic", "discord_webhook"] {
            assert!(!json.contains(forbidden), "settings must not contain {forbidden}");
        }
    }
}
