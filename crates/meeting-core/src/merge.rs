//! Merging the mic and desktop streams into one time-ordered transcript, and
//! rendering it as text.

use crate::labels::Renames;
use crate::segment::{Segment, Source};

/// Merge any number of per-stream segment lists into a single list ordered by
/// start time. Ties are broken by end time, then source (mic first), then id, so
/// the output is fully deterministic.
pub fn merge_segments<I>(streams: I) -> Vec<Segment>
where
    I: IntoIterator<Item = Vec<Segment>>,
{
    let mut all: Vec<Segment> = streams.into_iter().flatten().collect();
    sort_segments(&mut all);
    all
}

/// Sort in place using the same ordering as [`merge_segments`].
pub fn sort_segments(segments: &mut [Segment]) {
    segments.sort_by(|a, b| {
        a.start_ms
            .cmp(&b.start_ms)
            .then(a.end_ms.cmp(&b.end_ms))
            .then(source_rank(a.source).cmp(&source_rank(b.source)))
            .then(a.id.cmp(&b.id))
    });
}

fn source_rank(s: Source) -> u8 {
    match s {
        Source::Mic => 0,
        Source::Desktop => 1,
    }
}

/// `HH:MM:SS` from milliseconds.
pub fn format_timestamp(ms: u64) -> String {
    let total = ms / 1000;
    format!("{:02}:{:02}:{:02}", total / 3600, (total / 60) % 60, total % 60)
}

/// One rendered transcript line: `[00:01:23] Dana: hello there`.
pub fn render_line(seg: &Segment, renames: &Renames) -> String {
    format!(
        "[{}] {}: {}",
        format_timestamp(seg.start_ms),
        renames.apply(&seg.speaker),
        seg.text.trim()
    )
}

/// Join consecutive segments from the same (renamed) speaker when the gap
/// between them is at most `max_gap_ms`. This makes the transcript easier to
/// read and cheaper to send to the summarizer without losing any words.
pub fn coalesce_turns(segments: &[Segment], renames: &Renames, max_gap_ms: u64) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::with_capacity(segments.len());
    for seg in segments {
        let name = renames.apply(&seg.speaker);
        if let Some(last) = out.last_mut() {
            if renames.apply(&last.speaker) == name
                && seg.start_ms <= last.end_ms.saturating_add(max_gap_ms)
            {
                last.text = format!("{} {}", last.text.trim(), seg.text.trim());
                last.end_ms = last.end_ms.max(seg.end_ms);
                continue;
            }
        }
        out.push(seg.clone());
    }
    out
}

/// Render the full transcript as plain text (one line per turn), with renames applied.
pub fn render_transcript(segments: &[Segment], renames: &Renames) -> String {
    let mut s = String::new();
    for seg in segments {
        s.push_str(&render_line(seg, renames));
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(id: u64, source: Source, start: u64, end: u64, speaker: &str, text: &str) -> Segment {
        Segment {
            id,
            source,
            start_ms: start,
            end_ms: end,
            speaker: speaker.into(),
            text: text.into(),
            cluster: None,
        }
    }

    #[test]
    fn merges_two_streams_by_time() {
        let mic = vec![
            seg(1, Source::Mic, 1_000, 2_000, "Me", "hi"),
            seg(3, Source::Mic, 5_000, 6_000, "Me", "sounds good"),
        ];
        let desk = vec![
            seg(2, Source::Desktop, 2_500, 4_000, "Speaker 1", "hello, shall we start"),
            seg(4, Source::Desktop, 7_000, 8_000, "Speaker 2", "yes"),
        ];
        let merged = merge_segments(vec![mic, desk]);
        let ids: Vec<u64> = merged.iter().map(|s| s.id).collect();
        assert_eq!(ids, vec![1, 2, 3, 4]);
    }

    #[test]
    fn merge_handles_unsorted_input_and_ties_deterministically() {
        let a = vec![
            seg(10, Source::Desktop, 3_000, 4_000, "Speaker 1", "c"),
            seg(11, Source::Desktop, 1_000, 2_000, "Speaker 1", "a"),
        ];
        let b = vec![seg(12, Source::Mic, 1_000, 2_000, "Me", "b")];
        let merged = merge_segments(vec![a, b]);
        let ids: Vec<u64> = merged.iter().map(|s| s.id).collect();
        // Same start/end: mic sorts before desktop.
        assert_eq!(ids, vec![12, 11, 10]);
    }

    #[test]
    fn merge_with_one_empty_stream() {
        let merged = merge_segments(vec![vec![], vec![seg(1, Source::Mic, 0, 10, "Me", "x")]]);
        assert_eq!(merged.len(), 1);
        assert!(merge_segments(Vec::<Vec<Segment>>::new()).is_empty());
    }

    #[test]
    fn timestamps_format() {
        assert_eq!(format_timestamp(0), "00:00:00");
        assert_eq!(format_timestamp(83_999), "00:01:23");
        assert_eq!(format_timestamp(3_723_000), "01:02:03");
    }

    #[test]
    fn render_applies_renames() {
        let mut r = Renames::default();
        r.set("Speaker 2", "Dana");
        let segs = vec![
            seg(1, Source::Desktop, 61_000, 62_000, "Speaker 2", " Let's ship it. "),
            seg(2, Source::Mic, 63_000, 64_000, "Me", "Agreed"),
        ];
        let txt = render_transcript(&segs, &r);
        assert_eq!(txt, "[00:01:01] Dana: Let's ship it.\n[00:01:03] Me: Agreed\n");
    }

    #[test]
    fn coalesce_joins_same_speaker_with_small_gap() {
        let r = Renames::default();
        let segs = vec![
            seg(1, Source::Desktop, 0, 1_000, "Speaker 1", "one"),
            seg(2, Source::Desktop, 1_500, 2_000, "Speaker 1", "two"),
            seg(3, Source::Desktop, 9_000, 10_000, "Speaker 1", "three"),
            seg(4, Source::Mic, 10_100, 11_000, "Me", "four"),
        ];
        let out = coalesce_turns(&segs, &r, 2_000);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].text, "one two");
        assert_eq!(out[0].end_ms, 2_000);
        assert_eq!(out[1].text, "three");
    }

    #[test]
    fn coalesce_uses_renamed_identity() {
        // Two raw labels renamed to the same person become one turn.
        let mut r = Renames::default();
        r.set("Mic Speaker 1", "Alex");
        r.set("Speaker 3", "Alex");
        let segs = vec![
            seg(1, Source::Mic, 0, 1_000, "Mic Speaker 1", "a"),
            seg(2, Source::Desktop, 1_200, 2_000, "Speaker 3", "b"),
        ];
        assert_eq!(coalesce_turns(&segs, &r, 1_000).len(), 1);
    }
}
