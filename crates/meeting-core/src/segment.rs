use serde::{Deserialize, Serialize};

/// Which capture stream a segment came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    /// The local microphone.
    Mic,
    /// WASAPI loopback of the selected output device (the other participants).
    Desktop,
}

impl Source {
    pub fn as_str(&self) -> &'static str {
        match self {
            Source::Mic => "mic",
            Source::Desktop => "desktop",
        }
    }
}

/// One finalized, transcribed utterance.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    /// Monotonic id, unique within a recording session.
    pub id: u64,
    pub source: Source,
    /// Start time in milliseconds since the recording started (both streams share this clock).
    pub start_ms: u64,
    pub end_ms: u64,
    /// Raw speaker label *before* user renames, e.g. "Speaker 2", "Mic Speaker 1", "Dana".
    pub speaker: String,
    pub text: String,
    /// Diarization cluster index within its source (None when diarization did not run).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster: Option<usize>,
}
