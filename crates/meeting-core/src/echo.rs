//! Mic echo suppression.
//!
//! When the other participants come out of laptop speakers, the mic hears
//! them too and the same sentence is transcribed twice (once per stream). We
//! drop a mic segment when it overlaps in time with a desktop segment and
//! their words mostly match.

use std::collections::HashSet;

use crate::segment::{Segment, Source};

/// Time slack (ms) when comparing segment intervals.
const OVERLAP_SLACK_MS: u64 = 1_500;
/// Word-set overlap above which two segments are considered the same speech.
const SIMILARITY: f32 = 0.6;

fn words(s: &str) -> HashSet<String> {
    s.split_whitespace()
        .map(|w| w.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Overlap coefficient |A∩B| / min(|A|,|B|) — robust when one side is a fragment of the other.
pub fn word_similarity(a: &str, b: &str) -> f32 {
    let (wa, wb) = (words(a), words(b));
    if wa.is_empty() || wb.is_empty() {
        return 0.0;
    }
    let inter = wa.intersection(&wb).count() as f32;
    inter / wa.len().min(wb.len()) as f32
}

fn overlaps(a: &Segment, b: &Segment) -> bool {
    a.start_ms <= b.end_ms + OVERLAP_SLACK_MS && b.start_ms <= a.end_ms + OVERLAP_SLACK_MS
}

/// True if `mic` looks like an echo of `desktop`.
pub fn is_echo(mic: &Segment, desktop: &Segment) -> bool {
    mic.source == Source::Mic
        && desktop.source == Source::Desktop
        && overlaps(mic, desktop)
        && words(&mic.text).len() >= 2
        && word_similarity(&mic.text, &desktop.text) >= SIMILARITY
}

/// Remove mic segments that echo any desktop segment. Returns (kept, removed ids).
pub fn remove_mic_echoes(segments: &[Segment]) -> (Vec<Segment>, Vec<u64>) {
    let desktop: Vec<&Segment> = segments.iter().filter(|s| s.source == Source::Desktop).collect();
    let mut kept = Vec::with_capacity(segments.len());
    let mut removed = Vec::new();
    for s in segments {
        if s.source == Source::Mic && desktop.iter().any(|d| is_echo(s, d)) {
            removed.push(s.id);
        } else {
            kept.push(s.clone());
        }
    }
    (kept, removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(id: u64, source: Source, start: u64, end: u64, text: &str) -> Segment {
        Segment { id, source, start_ms: start, end_ms: end, speaker: String::new(), text: text.into(), cluster: None }
    }

    #[test]
    fn echo_is_detected_and_removed() {
        let segs = vec![
            seg(1, Source::Desktop, 1_000, 4_000, "We should ship the beta on Friday."),
            seg(2, Source::Mic, 1_200, 4_300, "we should ship the beta on friday"),
            seg(3, Source::Mic, 6_000, 7_000, "Sounds good to me."),
        ];
        let (kept, removed) = remove_mic_echoes(&segs);
        assert_eq!(removed, vec![2]);
        assert_eq!(kept.iter().map(|s| s.id).collect::<Vec<_>>(), vec![1, 3]);
    }

    #[test]
    fn same_words_far_apart_are_not_echo() {
        let d = seg(1, Source::Desktop, 1_000, 2_000, "yes that works");
        let m = seg(2, Source::Mic, 60_000, 61_000, "yes that works");
        assert!(!is_echo(&m, &d));
    }

    #[test]
    fn different_words_are_not_echo() {
        let d = seg(1, Source::Desktop, 1_000, 3_000, "what is the budget for the quarter");
        let m = seg(2, Source::Mic, 1_500, 3_000, "I can send the slides later");
        assert!(!is_echo(&m, &d));
    }

    #[test]
    fn single_word_is_never_echo() {
        let d = seg(1, Source::Desktop, 1_000, 2_000, "right");
        let m = seg(2, Source::Mic, 1_000, 2_000, "right");
        assert!(!is_echo(&m, &d));
    }

    #[test]
    fn similarity_bounds() {
        assert_eq!(word_similarity("", "a b"), 0.0);
        assert!((word_similarity("a b c", "A, B; C!") - 1.0).abs() < 1e-6);
    }
}
