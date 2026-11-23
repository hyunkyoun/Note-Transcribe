//! Settings held in memory, saved debounced on every change and synchronously on exit.

use std::path::PathBuf;
use std::time::Duration;

use crossbeam_channel::{RecvTimeoutError, Sender};
use meeting_core::settings::{load_settings, save_settings, LoadOutcome, Settings};
use parking_lot::Mutex;

const DEBOUNCE: Duration = Duration::from_millis(500);

pub struct SettingsStore {
    path: PathBuf,
    current: Mutex<Settings>,
    tx: Sender<Settings>,
    /// Human-readable note if the file was missing fields or corrupt.
    pub load_warning: Option<String>,
}

impl SettingsStore {
    pub fn load(path: PathBuf) -> Self {
        let (settings, outcome) = load_settings(&path);
        let load_warning = match &outcome {
            LoadOutcome::Loaded => None,
            LoadOutcome::Missing => None,
            LoadOutcome::Partial { dropped_fields } => Some(format!(
                "Some settings were invalid and reset to defaults: {}",
                dropped_fields.join(", ")
            )),
            LoadOutcome::Corrupt { error, backup } => Some(format!(
                "settings.json was unreadable ({error}); using defaults.{}",
                backup.as_ref().map(|b| format!(" The old file was moved to {}", b.display())).unwrap_or_default()
            )),
        };
        log::info!("settings: {:?} from {}", outcome, path.display());

        let (tx, rx) = crossbeam_channel::unbounded::<Settings>();
        let save_path = path.clone();
        std::thread::Builder::new()
            .name("settings-saver".into())
            .spawn(move || {
                while let Ok(mut latest) = rx.recv() {
                    // Wait until changes stop arriving for DEBOUNCE.
                    loop {
                        match rx.recv_timeout(DEBOUNCE) {
                            Ok(newer) => latest = newer,
                            Err(RecvTimeoutError::Timeout) => break,
                            Err(RecvTimeoutError::Disconnected) => break,
                        }
                    }
                    if let Err(e) = save_settings(&save_path, &latest) {
                        log::error!("saving settings failed: {e}");
                    }
                }
            })
            .expect("spawn settings saver");

        let store = Self { path, current: Mutex::new(settings), tx, load_warning };
        if matches!(outcome, LoadOutcome::Missing | LoadOutcome::Corrupt { .. } | LoadOutcome::Partial { .. }) {
            store.save_now(); // write a clean file right away
        }
        store
    }

    pub fn get(&self) -> Settings {
        self.current.lock().clone()
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    /// Replace settings (sanitized) and schedule a debounced save.
    pub fn update(&self, new: Settings) -> Settings {
        let s = new.sanitized();
        *self.current.lock() = s.clone();
        let _ = self.tx.send(s.clone());
        s
    }

    /// Synchronous save (window close / app exit).
    pub fn save_now(&self) {
        let s = self.get();
        if let Err(e) = save_settings(&self.path, &s) {
            log::error!("saving settings failed: {e}");
        }
    }
}
