//! Cleaning Whisper output and rejecting typical hallucinations.
//!
//! VAD already keeps silence away from Whisper; this is the second line of
//! defence for noise/music segments, where Whisper tends to emit stock phrases
//! ("Thanks for watching!") or loop on one word.

/// Phrases that are (almost) never real meeting speech when they make up a whole segment.
const STRONG: &[&str] = &[
    "thanks for watching",
    "thank you for watching",
    "thanks for watching and see you next time",
    "please subscribe",
    "like and subscribe",
    "dont forget to like and subscribe",
    "subtitles by the amaraorg community",
    "subtitles by",
    "transcribed by",
    "transcription by castingwords",
    "translated by",
    "copyright",
];

/// Phrases that are plausible speech but also common hallucinations; only
/// rejected when Whisper itself thinks the segment is probably not speech.
const WEAK: &[&str] = &["thank you", "thanks", "you", "bye", "bye bye", "okay", "oh", "uh", "um", "hmm", "so"];

fn normalize(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Remove `[BLANK_AUDIO]`, `[Music]`, `(applause)`-style annotations.
fn strip_annotations(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth_sq = 0usize;
    let mut depth_par = 0usize;
    let mut buf_par = String::new();
    for c in s.chars() {
        match c {
            '[' => depth_sq += 1,
            ']' if depth_sq > 0 => depth_sq -= 1,
            _ if depth_sq > 0 => {}
            '(' => {
                depth_par += 1;
                buf_par.push(c);
            }
            ')' if depth_par > 0 => {
                depth_par -= 1;
                buf_par.push(c);
                if depth_par == 0 {
                    // Keep parentheticals that look like real words of speech (> 3 words).
                    if buf_par.split_whitespace().count() > 3 {
                        out.push_str(&buf_par);
                    }
                    buf_par.clear();
                }
            }
            _ if depth_par > 0 => buf_par.push(c),
            '♪' | '♫' => {}
            _ => out.push(c),
        }
    }
    out.push_str(&buf_par);
    out
}

/// Collapse a word or short phrase repeated 4+ times in a row to one occurrence.
fn collapse_loops(words: Vec<String>) -> Vec<String> {
    let mut w = words;
    for n in 1..=4 {
        let mut out: Vec<String> = Vec::with_capacity(w.len());
        let mut i = 0;
        while i < w.len() {
            if i + n <= w.len() {
                let gram: Vec<String> = w[i..i + n].iter().map(|x| normalize(x)).collect();
                let mut reps = 1;
                while i + (reps + 1) * n <= w.len()
                    && w[i + reps * n..i + (reps + 1) * n].iter().map(|x| normalize(x)).collect::<Vec<_>>() == gram
                {
                    reps += 1;
                }
                if reps >= 4 {
                    out.extend_from_slice(&w[i..i + n]);
                    i += reps * n;
                    continue;
                }
            }
            out.push(w[i].clone());
            i += 1;
        }
        w = out;
    }
    w
}

/// Clean a Whisper segment. Returns `None` if it should be discarded.
///
/// `no_speech_prob` is Whisper's own probability that the segment has no speech.
pub fn clean_segment_text(text: &str, no_speech_prob: f32) -> Option<String> {
    let stripped = strip_annotations(text);
    let words: Vec<String> = stripped.split_whitespace().map(str::to_string).collect();
    let words = collapse_loops(words);
    let cleaned = words.join(" ").trim().to_string();
    if !cleaned.chars().any(char::is_alphanumeric) {
        return None;
    }
    if no_speech_prob > 0.8 {
        return None;
    }
    let norm = normalize(&cleaned);
    if STRONG.iter().any(|p| norm == *p || (norm.starts_with(p) && norm.len() < p.len() + 25)) {
        return None;
    }
    if no_speech_prob > 0.4 && WEAK.contains(&norm.as_str()) {
        return None;
    }
    Some(cleaned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_normal_speech() {
        assert_eq!(
            clean_segment_text("  Let's move the launch to Friday. ", 0.01).as_deref(),
            Some("Let's move the launch to Friday.")
        );
    }

    #[test]
    fn drops_annotations_only() {
        assert_eq!(clean_segment_text("[BLANK_AUDIO]", 0.0), None);
        assert_eq!(clean_segment_text(" (music) ♪ ", 0.0), None);
        assert_eq!(clean_segment_text("...", 0.0), None);
    }

    #[test]
    fn strips_annotations_inside_text() {
        assert_eq!(clean_segment_text("[Music] OK so the plan", 0.0).as_deref(), Some("OK so the plan"));
    }

    #[test]
    fn drops_stock_hallucinations() {
        assert_eq!(clean_segment_text("Thanks for watching!", 0.0), None);
        assert_eq!(clean_segment_text("Subtitles by the Amara.org community", 0.1), None);
    }

    #[test]
    fn weak_phrases_depend_on_no_speech_prob() {
        assert_eq!(clean_segment_text("Thank you.", 0.05).as_deref(), Some("Thank you."));
        assert_eq!(clean_segment_text("Thank you.", 0.6), None);
    }

    #[test]
    fn very_high_no_speech_is_dropped() {
        assert_eq!(clean_segment_text("Some words here", 0.95), None);
    }

    #[test]
    fn collapses_repetition_loops() {
        assert_eq!(clean_segment_text("you you you you you you", 0.0).as_deref(), Some("you"));
        assert_eq!(
            clean_segment_text("I think I think I think I think we should", 0.0).as_deref(),
            Some("I think we should")
        );
        // Three repeats is plausible speech and is kept.
        assert_eq!(clean_segment_text("no no no", 0.0).as_deref(), Some("no no no"));
    }
}
