//! Tiny file + stderr logger. Secrets are never passed to it; [`redact_url`]
//! exists for the rare case a URL must be mentioned.

use std::fs::{File, OpenOptions};
use std::io::Write;

use log::{LevelFilter, Metadata, Record};
use parking_lot::Mutex;

struct Logger {
    file: Mutex<Option<File>>,
}

impl log::Log for Logger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        // Keep our crate at info, silence chatty dependencies.
        metadata.target().starts_with("note_transcribe") || metadata.level() <= log::Level::Warn
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let line = format!(
            "{} {:<5} {}: {}\n",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
            record.level(),
            record.target(),
            record.args()
        );
        eprint!("{line}");
        if let Some(f) = self.file.lock().as_mut() {
            let _ = f.write_all(line.as_bytes());
        }
    }

    fn flush(&self) {
        if let Some(f) = self.file.lock().as_mut() {
            let _ = f.flush();
        }
    }
}

pub fn init() {
    let dir = crate::paths::logs_dir();
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("app.log");
    // Start a fresh log when the old one grows past 5 MB.
    if std::fs::metadata(&path).map(|m| m.len() > 5_000_000).unwrap_or(false) {
        let _ = std::fs::rename(&path, dir.join("app.old.log"));
    }
    let file = OpenOptions::new().create(true).append(true).open(&path).ok();
    let logger = Box::leak(Box::new(Logger { file: Mutex::new(file) }));
    if log::set_logger(logger).is_ok() {
        log::set_max_level(LevelFilter::Info);
    }
}

/// `https://discord.com/api/webhooks/123/abc` -> `https://discord.com/…`
pub fn redact_url(url: &str) -> String {
    match url.find("://") {
        Some(i) => {
            let rest = &url[i + 3..];
            let host = rest.split('/').next().unwrap_or("");
            format!("{}://{}/…", &url[..i], host)
        }
        None => "<redacted>".into(),
    }
}
