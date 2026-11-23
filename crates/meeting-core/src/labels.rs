//! Speaker labelling rules and the user's rename table.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::segment::{Segment, Source};

pub const DESKTOP_SPEAKER_PREFIX: &str = "Speaker";
pub const MIC_SPEAKER_PREFIX: &str = "Mic Speaker";
pub const DEFAULT_SELF_LABEL: &str = "Me";

/// Label used for mic segments when "Mic is only me" is on.
pub fn self_label(my_name: &str) -> String {
    let n = my_name.trim();
    if n.is_empty() {
        DEFAULT_SELF_LABEL.to_string()
    } else {
        n.to_string()
    }
}

/// Whether diarization should run for a given source.
///
/// Desktop audio is always diarized. Mic audio is diarized only when the user
/// has said other people may be speaking into the mic ("Mic is only me" = off).
pub fn should_diarize(source: Source, mic_is_only_me: bool) -> bool {
    match source {
        Source::Desktop => true,
        Source::Mic => !mic_is_only_me,
    }
}

/// Raw label for a segment.
///
/// * Mic + "only me" on  -> my name (or "Me").
/// * Mic + "only me" off -> "Mic Speaker N" (kept distinct from desktop speakers).
/// * Desktop             -> "Speaker N".
///
/// `cluster` is the 0-based diarization cluster; `None` (diarization
/// unavailable / segment too short with no previous speaker) maps to cluster 0.
pub fn label_for(source: Source, mic_is_only_me: bool, my_name: &str, cluster: Option<usize>) -> String {
    let n = cluster.unwrap_or(0) + 1;
    match source {
        Source::Mic if mic_is_only_me => self_label(my_name),
        Source::Mic => format!("{MIC_SPEAKER_PREFIX} {n}"),
        Source::Desktop => format!("{DESKTOP_SPEAKER_PREFIX} {n}"),
    }
}

/// User-defined renames, e.g. `"Speaker 2" -> "Dana"`. Applied to the whole
/// transcript and to everything sent to Claude and Discord.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Renames(pub BTreeMap<String, String>);

impl Renames {
    pub fn set(&mut self, from: &str, to: &str) {
        self.0.insert(from.to_string(), to.to_string());
    }

    /// Display name for a raw label. Blank renames are ignored.
    pub fn apply(&self, label: &str) -> String {
        match self.0.get(label) {
            Some(to) if !to.trim().is_empty() => to.trim().to_string(),
            _ => label.to_string(),
        }
    }
}

/// Return copies of the segments with renames baked into `speaker`.
pub fn apply_renames(segments: &[Segment], renames: &Renames) -> Vec<Segment> {
    segments
        .iter()
        .map(|s| Segment {
            speaker: renames.apply(&s.speaker),
            ..s.clone()
        })
        .collect()
}

/// Unique raw speaker labels in order of first appearance.
pub fn unique_speakers(segments: &[Segment]) -> Vec<String> {
    let mut seen = Vec::<String>::new();
    for s in segments {
        if !seen.iter().any(|x| x == &s.speaker) {
            seen.push(s.speaker.clone());
        }
    }
    seen
}

/// Unique display names (after renames) in order of first appearance.
pub fn unique_display_names(segments: &[Segment], renames: &Renames) -> Vec<String> {
    let mut seen = Vec::<String>::new();
    for s in segments {
        let n = renames.apply(&s.speaker);
        if !seen.contains(&n) {
            seen.push(n);
        }
    }
    seen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mic_only_me_on_uses_name() {
        assert_eq!(label_for(Source::Mic, true, "Eun", Some(3)), "Eun");
        assert_eq!(label_for(Source::Mic, true, "  Eun  ", None), "Eun");
    }

    #[test]
    fn mic_only_me_on_empty_name_is_me() {
        assert_eq!(label_for(Source::Mic, true, "", None), "Me");
        assert_eq!(label_for(Source::Mic, true, "   ", Some(1)), "Me");
    }

    #[test]
    fn mic_only_me_off_uses_mic_speaker_clusters() {
        assert_eq!(label_for(Source::Mic, false, "Eun", Some(0)), "Mic Speaker 1");
        assert_eq!(label_for(Source::Mic, false, "Eun", Some(2)), "Mic Speaker 3");
        assert_eq!(label_for(Source::Mic, false, "", None), "Mic Speaker 1");
    }

    #[test]
    fn desktop_labels_ignore_mic_toggle() {
        for toggle in [true, false] {
            assert_eq!(label_for(Source::Desktop, toggle, "Eun", Some(1)), "Speaker 2");
        }
    }

    #[test]
    fn diarize_decision() {
        assert!(should_diarize(Source::Desktop, true));
        assert!(should_diarize(Source::Desktop, false));
        assert!(!should_diarize(Source::Mic, true));
        assert!(should_diarize(Source::Mic, false));
    }

    #[test]
    fn mic_and_desktop_labels_never_collide() {
        let mic = label_for(Source::Mic, false, "", Some(0));
        let desk = label_for(Source::Desktop, false, "", Some(0));
        assert_ne!(mic, desk);
    }

    #[test]
    fn renames_apply_and_ignore_blank() {
        let mut r = Renames::default();
        r.set("Speaker 2", " Dana ");
        r.set("Speaker 3", "   ");
        assert_eq!(r.apply("Speaker 2"), "Dana");
        assert_eq!(r.apply("Speaker 3"), "Speaker 3");
        assert_eq!(r.apply("Speaker 9"), "Speaker 9");
    }

    #[test]
    fn unique_speakers_in_order() {
        let mk = |sp: &str| Segment {
            id: 0,
            source: Source::Desktop,
            start_ms: 0,
            end_ms: 0,
            speaker: sp.into(),
            text: String::new(),
            cluster: None,
        };
        let segs = vec![mk("Speaker 2"), mk("Me"), mk("Speaker 2"), mk("Speaker 1")];
        assert_eq!(unique_speakers(&segs), vec!["Speaker 2", "Me", "Speaker 1"]);
        let mut r = Renames::default();
        r.set("Speaker 1", "Dana");
        r.set("Speaker 2", "Dana");
        assert_eq!(unique_display_names(&segs, &r), vec!["Dana", "Me"]);
        let renamed = apply_renames(&segs, &r);
        assert_eq!(renamed[0].speaker, "Dana");
        assert_eq!(renamed[1].speaker, "Me");
    }
}
