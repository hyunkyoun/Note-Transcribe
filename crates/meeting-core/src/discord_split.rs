//! Splitting long text into Discord-sized messages.
//!
//! Discord rejects message `content` longer than 2000 characters. We split on
//! the most natural boundary available (blank line, then line, then sentence,
//! then word, then a hard character split) and keep ``` code fences balanced
//! across chunks so formatting survives.

/// Discord's per-message content limit.
pub const DISCORD_LIMIT: usize = 2000;

const FENCE: &str = "```";
/// Room reserved for closing/reopening a code fence across a split.
const FENCE_RESERVE: usize = 8;

fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// Byte index of the `n`-th char (or len if fewer chars).
fn byte_at_char(s: &str, n: usize) -> usize {
    s.char_indices().nth(n).map(|(i, _)| i).unwrap_or(s.len())
}

/// Find a split point (byte index) no later than `max_chars` characters into `s`.
fn find_split(s: &str, max_chars: usize) -> usize {
    let hard = byte_at_char(s, max_chars);
    let window = &s[..hard];
    // Don't accept a split that leaves a tiny first chunk; prefer at least 30% full.
    let min = window.len() * 3 / 10;
    for pat in ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "] {
        if let Some(i) = window.rfind(pat) {
            let cut = i + pat.len();
            if cut > min && cut <= window.len() {
                return cut;
            }
        }
    }
    hard.max(s.char_indices().nth(1).map(|(i, _)| i).unwrap_or(s.len()))
}

/// Split `text` into chunks of at most `limit` characters each.
pub fn split_message(text: &str, limit: usize) -> Vec<String> {
    assert!(limit > FENCE_RESERVE * 2, "limit too small");
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut rest: String = text.to_string();
    let mut reopen_fence = false;

    loop {
        let prefix = if reopen_fence { format!("{FENCE}\n") } else { String::new() };
        let budget = limit - char_len(&prefix);
        if char_len(&rest) <= budget {
            let chunk = format!("{prefix}{rest}");
            if !chunk.trim().is_empty() {
                out.push(chunk);
            }
            break;
        }
        let cut = find_split(&rest, budget - FENCE_RESERVE);
        let head = rest[..cut].trim_end().to_string();
        let tail = rest[cut..].trim_start_matches([' ', '\n', '\r']).to_string();

        let mut chunk = format!("{prefix}{head}");
        // An odd number of fences means we are inside a code block at the cut.
        let inside = chunk.matches(FENCE).count() % 2 == 1;
        if inside {
            chunk.push('\n');
            chunk.push_str(FENCE);
        }
        out.push(chunk);
        reopen_fence = inside;
        rest = tail;
        if rest.is_empty() {
            break;
        }
    }
    out
}

/// Render a transcript for Discord: paste it in a code block if short enough,
/// otherwise the caller should attach it as a file.
pub fn transcript_fits_inline(transcript: &str, limit: usize) -> bool {
    // "```\n" + text + "\n```" plus a short header line.
    char_len(transcript) + 40 <= limit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_single_chunk() {
        assert_eq!(split_message("hello", DISCORD_LIMIT), vec!["hello"]);
        assert!(split_message("   ", DISCORD_LIMIT).is_empty());
    }

    #[test]
    fn every_chunk_within_limit_and_content_preserved() {
        let para = "This is a sentence about the roadmap. ".repeat(40);
        let text = format!("# Summary\n\n{para}\n\n## Action items\n\n{para}\n\n{para}");
        let chunks = split_message(&text, DISCORD_LIMIT);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(c.chars().count() <= DISCORD_LIMIT, "chunk too long: {}", c.chars().count());
        }
        let rejoined: String = chunks.join(" ").split_whitespace().collect::<Vec<_>>().join(" ");
        let original: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
        assert_eq!(rejoined, original);
    }

    #[test]
    fn prefers_paragraph_boundaries() {
        let a = "a".repeat(1200);
        let b = "b".repeat(1200);
        let chunks = split_message(&format!("{a}\n\n{b}"), DISCORD_LIMIT);
        assert_eq!(chunks, vec![a, b]);
    }

    #[test]
    fn hard_splits_unbroken_text() {
        let s = "x".repeat(4500);
        let chunks = split_message(&s, DISCORD_LIMIT);
        assert_eq!(chunks.len(), 3);
        assert!(chunks.iter().all(|c| c.chars().count() <= DISCORD_LIMIT));
        assert_eq!(chunks.concat().len(), 4500);
    }

    #[test]
    fn multibyte_characters_are_not_broken() {
        let s = "é🙂漢".repeat(1000); // 3000 chars, many bytes
        let chunks = split_message(&s, DISCORD_LIMIT);
        assert!(chunks.iter().all(|c| c.chars().count() <= DISCORD_LIMIT));
        assert_eq!(chunks.concat(), s);
    }

    #[test]
    fn code_fences_stay_balanced() {
        let code = (0..300).map(|i| format!("line {i}\n")).collect::<String>();
        let text = format!("Intro\n```\n{code}```\nOutro");
        let chunks = split_message(&text, DISCORD_LIMIT);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert_eq!(c.matches("```").count() % 2, 0, "unbalanced fence in chunk: {c}");
            assert!(c.chars().count() <= DISCORD_LIMIT);
        }
    }

    #[test]
    fn inline_threshold() {
        assert!(transcript_fits_inline("short", DISCORD_LIMIT));
        assert!(!transcript_fits_inline(&"x".repeat(1990), DISCORD_LIMIT));
    }
}
