use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

use crate::diarize::SharedExtractor;
use crate::meeting::Meeting;
use crate::session::ActiveRecording;
use crate::settings_store::SettingsStore;
use crate::transcriber::EngineInfo;
use crate::worker::EngineSlot;

pub struct AppState {
    pub settings: SettingsStore,
    pub recording: Mutex<Option<ActiveRecording>>,
    pub meeting: Mutex<Option<Meeting>>,
    pub engine: EngineSlot,
    pub extractor: Mutex<Option<SharedExtractor>>,
    pub last_engine: Mutex<Option<EngineInfo>>,
    pub http: reqwest::Client,
    /// A summary / Discord job is running.
    pub busy: AtomicBool,
}

impl AppState {
    pub fn new(settings: SettingsStore) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(20 * 60))
            .user_agent(concat!("note-transcribe/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("http client");
        Self {
            settings,
            recording: Mutex::new(None),
            meeting: Mutex::new(Meeting::load_last()),
            engine: Arc::new(Mutex::new(None)),
            extractor: Mutex::new(None),
            last_engine: Mutex::new(None),
            http,
            busy: AtomicBool::new(false),
        }
    }
}
