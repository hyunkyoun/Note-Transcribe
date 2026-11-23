//! Secrets and API configuration from `.env`. Never logged, never written to settings.json.

use std::fmt;
use std::path::PathBuf;

use parking_lot::Mutex;
use serde::Serialize;

use crate::paths;

static LOADED_ENV: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Load `.env` (first match: data dir, then exe/cwd/resource search). Values
/// already set in the process environment win. Re-callable ("Reload .env").
pub fn load_env() -> Option<PathBuf> {
    let mut candidates = vec![paths::data_dir().join(".env")];
    if let Some(p) = paths::find_project_path(".env") {
        candidates.push(p);
    }
    for p in candidates {
        if p.is_file() {
            // `from_path_override` so edits to .env take effect on reload.
            match dotenvy::from_path_override(&p) {
                Ok(()) => {
                    log::info!("loaded environment from {}", p.display());
                    *LOADED_ENV.lock() = Some(p.clone());
                    return Some(p);
                }
                Err(e) => log::warn!("failed to parse {}: {e}", p.display()),
            }
        }
    }
    log::warn!("no .env file found; API features disabled until one is created");
    *LOADED_ENV.lock() = None;
    None
}

pub fn loaded_env_path() -> Option<PathBuf> {
    LOADED_ENV.lock().clone()
}

/// Treat empty values and the placeholders shipped in `.env.example` as unset.
fn secret(name: &str) -> Option<String> {
    let v = std::env::var(name).ok()?;
    let v = v.trim().trim_matches('"').trim().to_string();
    let lower = v.to_lowercase();
    if v.is_empty()
        || lower.contains("your-")
        || lower.contains("your_")
        || lower.contains("replace-me")
        || lower.contains("changeme")
        || lower == "xxx"
    {
        None
    } else {
        Some(v)
    }
}

fn plain(name: &str) -> Option<String> {
    std::env::var(name).ok().map(|v| v.trim().trim_matches('"').trim().to_string()).filter(|v| !v.is_empty())
}

pub const DEFAULT_CLAUDE_MODEL: &str = "claude-opus-5-5";

#[derive(Clone)]
pub struct ApiConfig {
    pub anthropic_api_key: Option<String>,
    pub claude_model: String,
    pub claude_max_tokens: u32,
    /// Optional `output_config.effort` (low/medium/high/xhigh/max). Unset = model default.
    pub claude_effort: Option<String>,
    /// Above this estimated input size the transcript is summarized in chunks.
    pub claude_max_input_tokens: usize,
    /// Server-side refusal fallbacks ("default"/"off").
    pub claude_fallbacks: bool,
    pub anthropic_base_url: String,
    pub discord_webhook_url: Option<String>,
    pub hf_token: Option<String>,
}

impl ApiConfig {
    /// Read the current environment (call after [`load_env`]).
    pub fn from_env() -> Self {
        Self {
            anthropic_api_key: secret("ANTHROPIC_API_KEY"),
            claude_model: plain("CLAUDE_MODEL").unwrap_or_else(|| DEFAULT_CLAUDE_MODEL.to_string()),
            claude_max_tokens: plain("CLAUDE_MAX_TOKENS").and_then(|v| v.parse().ok()).unwrap_or(16_000),
            claude_effort: plain("CLAUDE_EFFORT").map(|s| s.to_lowercase()),
            claude_max_input_tokens: plain("CLAUDE_MAX_INPUT_TOKENS")
                .and_then(|v| v.parse().ok())
                .unwrap_or(150_000),
            claude_fallbacks: plain("CLAUDE_FALLBACKS").map(|v| v.to_lowercase() != "off").unwrap_or(true),
            anthropic_base_url: plain("ANTHROPIC_BASE_URL").unwrap_or_else(|| "https://api.anthropic.com".into()),
            discord_webhook_url: secret("DISCORD_WEBHOOK_URL"),
            hf_token: secret("HF_TOKEN"),
        }
    }

    pub fn status(&self) -> ConfigStatus {
        ConfigStatus {
            has_anthropic_key: self.anthropic_api_key.is_some(),
            claude_model: self.claude_model.clone(),
            has_discord_webhook: self.discord_webhook_url.is_some(),
            env_path: loaded_env_path().map(|p| p.display().to_string()),
        }
    }
}

/// Redacting Debug so a stray `{:?}` can never leak a secret into logs.
impl fmt::Debug for ApiConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApiConfig")
            .field("anthropic_api_key", &self.anthropic_api_key.as_ref().map(|_| "<redacted>"))
            .field("claude_model", &self.claude_model)
            .field("claude_max_tokens", &self.claude_max_tokens)
            .field("claude_effort", &self.claude_effort)
            .field("claude_max_input_tokens", &self.claude_max_input_tokens)
            .field("claude_fallbacks", &self.claude_fallbacks)
            .field("discord_webhook_url", &self.discord_webhook_url.as_ref().map(|_| "<redacted>"))
            .field("hf_token", &self.hf_token.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ConfigStatus {
    pub has_anthropic_key: bool,
    pub claude_model: String,
    pub has_discord_webhook: bool,
    pub env_path: Option<String>,
}
