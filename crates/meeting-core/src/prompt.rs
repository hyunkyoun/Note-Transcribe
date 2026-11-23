//! Summary prompt templating and transcript chunking for long meetings.

use std::collections::HashMap;

/// Placeholders supported in `prompts/summary_prompt.md`.
pub const PH_TRANSCRIPT: &str = "TRANSCRIPT";
pub const PH_DATE: &str = "DATE";
pub const PH_DURATION: &str = "DURATION";
pub const PH_SPEAKERS: &str = "SPEAKERS";

/// Replace `{{NAME}}` (whitespace inside the braces allowed, name is
/// case-insensitive) with values from `vars`. Unknown placeholders are left
/// untouched. If the template has no `{{TRANSCRIPT}}` placeholder the
/// transcript is appended at the end so it is never silently dropped.
pub fn render_template(template: &str, vars: &HashMap<String, String>) -> String {
    let mut out = String::with_capacity(template.len() + vars.values().map(|v| v.len()).sum::<usize>());
    let mut rest = template;
    let mut saw_transcript = false;
    while let Some(open) = rest.find("{{") {
        out.push_str(&rest[..open]);
        let after = &rest[open + 2..];
        match after.find("}}") {
            Some(close) => {
                let name = after[..close].trim().to_ascii_uppercase();
                let valid = !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
                match vars.get(&name) {
                    Some(v) if valid => {
                        if name == PH_TRANSCRIPT {
                            saw_transcript = true;
                        }
                        out.push_str(v);
                    }
                    _ => out.push_str(&rest[open..open + 2 + close + 2]),
                }
                rest = &after[close + 2..];
            }
            None => {
                out.push_str(&rest[open..]);
                rest = "";
            }
        }
    }
    out.push_str(rest);
    if !saw_transcript {
        if let Some(t) = vars.get(PH_TRANSCRIPT) {
            out.push_str("\n\n<transcript>\n");
            out.push_str(t);
            out.push_str("\n</transcript>\n");
        }
    }
    out
}

/// Build the standard variable map.
pub fn standard_vars(transcript: &str, date: &str, duration: &str, speakers: &[String]) -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert(PH_TRANSCRIPT.to_string(), transcript.to_string());
    m.insert(PH_DATE.to_string(), date.to_string());
    m.insert(PH_DURATION.to_string(), duration.to_string());
    m.insert(PH_SPEAKERS.to_string(), speakers.join(", "));
    m
}

/// Human-readable duration, e.g. `1h 05m` or `42m 10s`.
pub fn format_duration(ms: u64) -> String {
    let s = ms / 1000;
    let (h, m, sec) = (s / 3600, (s / 60) % 60, s % 60);
    if h > 0 {
        format!("{h}h {m:02}m")
    } else {
        format!("{m}m {sec:02}s")
    }
}

/// Conservative token estimate (English averages ~4 chars/token; we assume 3.5
/// so we chunk a little early rather than overflow the context window).
pub fn estimate_tokens(text: &str) -> usize {
    (text.chars().count() as f64 / 3.5).ceil() as usize
}

/// Split transcript text into chunks of at most `max_chars` characters,
/// breaking only between lines where possible (a single over-long line is
/// split on a char boundary).
pub fn chunk_transcript(text: &str, max_chars: usize) -> Vec<String> {
    assert!(max_chars > 0);
    let mut chunks = Vec::new();
    let mut cur = String::new();
    let mut cur_len = 0usize;
    for line in text.lines() {
        let mut line = line.to_string();
        loop {
            let len = line.chars().count();
            if cur_len + len < max_chars {
                cur.push_str(&line);
                cur.push('\n');
                cur_len += len + 1;
                break;
            }
            if cur_len > 0 {
                chunks.push(std::mem::take(&mut cur));
                cur_len = 0;
                continue;
            }
            // Line alone is too long: hard split.
            let take = max_chars.saturating_sub(1).max(1);
            let idx = line.char_indices().nth(take).map(|(i, _)| i).unwrap_or(line.len());
            let (head, tail) = line.split_at(idx);
            chunks.push(format!("{head}\n"));
            line = tail.to_string();
            if line.is_empty() {
                break;
            }
        }
    }
    if cur_len > 0 {
        chunks.push(cur);
    }
    chunks
}

/// Instructions used for the per-chunk "notes" pass when a meeting is too long
/// to summarize in one request. The final pass then feeds these notes into the
/// user's own `summary_prompt.md`, so the final output format is still
/// controlled entirely by that file.
pub fn chunk_notes_prompt(index: usize, total: usize, chunk: &str) -> String {
    format!(
        "You are helping summarize a long meeting transcript that has been split into {total} parts. \
This is part {part} of {total}.\n\n\
Write detailed, faithful notes for this part only. Preserve: who said what (use the speaker names exactly as written), \
topics discussed, any decisions, every action item with its owner and due date if stated, open questions, numbers, dates and names. \
Include approximate timestamps for important moments. Do not invent anything that is not in the transcript. \
Do not write an introduction or conclusion; just the notes.\n\n<transcript_part>\n{chunk}</transcript_part>",
        part = index + 1
    )
}

/// Text substituted for `{{TRANSCRIPT}}` in the final pass of a chunked summary.
pub fn combined_notes_transcript(notes: &[String]) -> String {
    let mut s = String::from(
        "(This meeting was too long to send in one request. Below are detailed notes of each consecutive part of the transcript, in order. Treat them as the transcript.)\n\n",
    );
    for (i, n) in notes.iter().enumerate() {
        s.push_str(&format!("### Part {} of {}\n{}\n\n", i + 1, notes.len(), n.trim()));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars() -> HashMap<String, String> {
        standard_vars("[00:00:01] Me: hi", "2026-11-23", "5m 00s", &["Me".into(), "Dana".into()])
    }

    #[test]
    fn replaces_all_placeholders() {
        let t = "Date: {{DATE}} ({{ duration }})\nPeople: {{SPEAKERS}}\n---\n{{TRANSCRIPT}}";
        let out = render_template(t, &vars());
        assert_eq!(out, "Date: 2026-11-23 (5m 00s)\nPeople: Me, Dana\n---\n[00:00:01] Me: hi");
    }

    #[test]
    fn unknown_placeholders_are_left_alone() {
        let out = render_template("{{FOO}} {{TRANSCRIPT}} {{ }}", &vars());
        assert_eq!(out, "{{FOO}} [00:00:01] Me: hi {{ }}");
    }

    #[test]
    fn missing_transcript_placeholder_appends_transcript() {
        let out = render_template("Summarize this meeting.", &vars());
        assert!(out.starts_with("Summarize this meeting."));
        assert!(out.contains("<transcript>\n[00:00:01] Me: hi\n</transcript>"));
    }

    #[test]
    fn transcript_is_inserted_verbatim_even_with_braces() {
        let mut v = vars();
        v.insert("TRANSCRIPT".into(), "Me: {{DATE}} is literal".into());
        let out = render_template("{{TRANSCRIPT}}", &v);
        // Values are not re-expanded.
        assert_eq!(out, "Me: {{DATE}} is literal");
    }

    #[test]
    fn unterminated_braces_are_kept() {
        assert_eq!(render_template("a {{DATE", &vars()), "a {{DATE\n\n<transcript>\n[00:00:01] Me: hi\n</transcript>\n");
    }

    #[test]
    fn repeated_placeholder() {
        let out = render_template("{{DATE}}/{{DATE}} {{TRANSCRIPT}}", &vars());
        assert!(out.starts_with("2026-11-23/2026-11-23 "));
    }

    #[test]
    fn durations() {
        assert_eq!(format_duration(0), "0m 00s");
        assert_eq!(format_duration(65_000), "1m 05s");
        assert_eq!(format_duration(3_900_000), "1h 05m");
    }

    #[test]
    fn chunking_respects_limit_and_line_boundaries() {
        let text: String = (0..1000).map(|i| format!("[00:00:{:02}] Speaker 1: line number {i}\n", i % 60)).collect();
        let chunks = chunk_transcript(&text, 5_000);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(c.chars().count() <= 5_000);
            assert!(c.ends_with('\n'));
        }
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn chunking_splits_overlong_line() {
        let text = "y".repeat(25);
        let chunks = chunk_transcript(&text, 10);
        assert!(chunks.iter().all(|c| c.chars().count() <= 10));
        assert_eq!(chunks.concat().replace('\n', ""), text);
    }

    #[test]
    fn token_estimate_is_conservative() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens(&"a".repeat(35)), 10);
    }

    #[test]
    fn combined_notes_numbering() {
        let s = combined_notes_transcript(&["n1".into(), "n2".into()]);
        assert!(s.contains("### Part 1 of 2\nn1"));
        assert!(s.contains("### Part 2 of 2\nn2"));
        assert!(chunk_notes_prompt(1, 3, "x\n").contains("This is part 2 of 3"));
    }
}
