//! A finished meeting: segments, renames, summary, and its files on disk.
//!
//! Files written to the transcripts folder (all named after the session id,
//! e.g. `2026-11-23_1405`):
//! * `<id>.jsonl`       – crash log, appended live, one segment per line
//! * `<id>.txt`         – readable transcript (live, then rewritten with final labels/renames)
//! * `<id>.json`        – this struct (final segments, renames, summary)
//! * `<id>.summary.md`  – the Claude summary
//! * `<id>.mic.wav` / `<id>.desktop.wav` – only if "Save raw audio" is on

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use meeting_core::labels::{unique_display_names, unique_speakers, Renames};
use meeting_core::merge::{coalesce_turns, render_transcript};
use meeting_core::prompt::format_duration;
use meeting_core::Segment;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Meeting {
    pub id: String,
    /// Local start time, e.g. "2026-11-23 14:05".
    pub started_label: String,
    pub duration_ms: u64,
    pub dir: PathBuf,
    pub segments: Vec<Segment>,
    #[serde(default)]
    pub renames: Renames,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub discord_posted: bool,
}

/// What the UI needs to show the rename table and status.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MeetingView {
    pub id: String,
    pub started_label: String,
    pub duration: String,
    pub speakers: Vec<String>,
    pub renames: Renames,
    pub segment_count: usize,
    pub transcript_path: String,
    pub summary: Option<String>,
    pub summary_path: Option<String>,
    pub discord_posted: bool,
}

impl Meeting {
    pub fn path(&self, ext: &str) -> PathBuf {
        self.dir.join(format!("{}.{ext}", self.id))
    }

    pub fn json_path(&self) -> PathBuf {
        self.path("json")
    }
    pub fn txt_path(&self) -> PathBuf {
        self.path("txt")
    }
    pub fn summary_path(&self) -> PathBuf {
        self.path("summary.md")
    }

    /// Readable transcript with renames applied (one line per turn).
    pub fn transcript_text(&self) -> String {
        let turns = coalesce_turns(&self.segments, &self.renames, 1_500);
        render_transcript(&turns, &self.renames)
    }

    pub fn display_speakers(&self) -> Vec<String> {
        unique_display_names(&self.segments, &self.renames)
    }

    pub fn duration_label(&self) -> String {
        format_duration(self.duration_ms)
    }

    pub fn view(&self) -> MeetingView {
        MeetingView {
            id: self.id.clone(),
            started_label: self.started_label.clone(),
            duration: self.duration_label(),
            speakers: unique_speakers(&self.segments),
            renames: self.renames.clone(),
            segment_count: self.segments.len(),
            transcript_path: self.txt_path().display().to_string(),
            summary: self.summary.clone(),
            summary_path: self.summary.as_ref().map(|_| self.summary_path().display().to_string()),
            discord_posted: self.discord_posted,
        }
    }

    /// Write `<id>.json` and `<id>.txt` (and the summary if present).
    pub fn save(&self) -> Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let header = format!(
            "Meeting {} ({})\nSpeakers: {}\n\n",
            self.started_label,
            self.duration_label(),
            self.display_speakers().join(", ")
        );
        write_atomic(&self.txt_path(), &(header + &self.transcript_text()))?;
        write_atomic(&self.json_path(), &serde_json::to_string_pretty(self)?)?;
        if let Some(s) = &self.summary {
            write_atomic(&self.summary_path(), s)?;
        }
        std::fs::write(last_pointer(), self.json_path().display().to_string()).ok();
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        Ok(serde_json::from_str(&text)?)
    }

    /// The most recently finished meeting (survives restarts, for "Re-send").
    pub fn load_last() -> Option<Self> {
        let p = std::fs::read_to_string(last_pointer()).ok()?;
        Self::load(Path::new(p.trim())).ok()
    }
}

fn last_pointer() -> PathBuf {
    crate::paths::data_dir().join("last_meeting.txt")
}

pub fn write_atomic(path: &Path, content: &str) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, content)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}
