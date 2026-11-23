//! Transcription job queue and the single GPU worker thread.
//!
//! Both capture streams share one Whisper model. Final segments always run
//! before partial (live preview) jobs, and a newer partial for a stream
//! replaces an older one that hasn't started yet, so captions stay fresh
//! even when the GPU is briefly busy.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;

use meeting_core::hallucination::clean_segment_text;
use meeting_core::settings::Settings;
use meeting_core::{Segment, Source};
use parking_lot::{Condvar, Mutex};

use crate::events::{CaptionEvent, Events};
use crate::session::SessionShared;
use crate::transcriber::{DecodeOptions, WhisperEngine};

pub struct FinalJob {
    pub source: Source,
    pub id: u64,
    pub start_ms: u64,
    pub end_ms: u64,
    pub samples: Vec<f32>,
    pub speaker: String,
    pub cluster: Option<usize>,
}

pub struct PartialJob {
    pub source: Source,
    pub start_ms: u64,
    pub end_ms: u64,
    pub samples: Vec<f32>,
    pub speaker_hint: String,
}

pub enum Job {
    Final(FinalJob),
    Partial(PartialJob),
}

#[derive(Default)]
struct Inner {
    finals: VecDeque<FinalJob>,
    partials: HashMap<Source, PartialJob>,
    closed: bool,
}

#[derive(Default)]
pub struct JobQueue {
    inner: Mutex<Inner>,
    cv: Condvar,
}

impl JobQueue {
    pub fn push_final(&self, job: FinalJob) {
        let mut g = self.inner.lock();
        g.partials.remove(&job.source); // superseded
        g.finals.push_back(job);
        self.cv.notify_one();
    }

    pub fn push_partial(&self, job: PartialJob) {
        let mut g = self.inner.lock();
        if g.closed || g.finals.iter().any(|f| f.source == job.source) {
            return;
        }
        g.partials.insert(job.source, job);
        self.cv.notify_one();
    }

    /// No more jobs will be added; the worker exits once the queue is empty.
    pub fn close(&self, discard_pending: bool) {
        let mut g = self.inner.lock();
        g.closed = true;
        g.partials.clear();
        if discard_pending {
            g.finals.clear();
        }
        self.cv.notify_all();
    }

    pub fn pending_finals(&self) -> usize {
        self.inner.lock().finals.len()
    }

    fn pop(&self) -> Option<Job> {
        let mut g = self.inner.lock();
        loop {
            if let Some(f) = g.finals.pop_front() {
                return Some(Job::Final(f));
            }
            // Oldest partial first so neither stream starves.
            if let Some(src) = g.partials.iter().min_by_key(|(_, p)| p.end_ms).map(|(s, _)| *s) {
                return g.partials.remove(&src).map(Job::Partial);
            }
            if g.closed {
                return None;
            }
            self.cv.wait(&mut g);
        }
    }
}

pub type EngineSlot = Arc<Mutex<Option<WhisperEngine>>>;

pub fn spawn_worker(
    queue: Arc<JobQueue>,
    slot: EngineSlot,
    models_dir: PathBuf,
    settings: Settings,
    session: Arc<SessionShared>,
    events: Events,
) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name("whisper-worker".into())
        .spawn(move || {
            let cached = slot.lock().take();
            let mut engine = match cached {
                Some(e) => {
                    events.engine("ready", format!("Whisper ready ({})", e.info.model), Some(e.info.clone()));
                    e
                }
                None => {
                    events.engine("loading", "Loading Whisper model…", None);
                    match WhisperEngine::load(&models_dir, &settings) {
                        Ok(e) => {
                            let msg = format!(
                                "Whisper ready: {} on {}",
                                e.info.model,
                                e.info.gpu_name.clone().unwrap_or_else(|| "CPU".into())
                            );
                            events.engine("ready", msg, Some(e.info.clone()));
                            e
                        }
                        Err(err) => {
                            log::error!("whisper load failed: {err:#}");
                            events.engine("error", format!("Transcription unavailable: {err:#}"), None);
                            // Drain so producers never block; audio is still saved if enabled.
                            while queue.pop().is_some() {}
                            return;
                        }
                    }
                }
            };
            session.set_engine_info(engine.info.clone());

            let language = settings.language.clone();
            let prompt = settings.initial_prompt.clone();
            while let Some(job) = queue.pop() {
                match job {
                    Job::Final(j) => {
                        let opts = DecodeOptions { language: &language, initial_prompt: &prompt, partial: false };
                        let text = match engine.transcribe(&j.samples, &opts) {
                            Ok(segs) => segs
                                .iter()
                                .filter_map(|s| clean_segment_text(&s.text, s.no_speech))
                                .collect::<Vec<_>>()
                                .join(" "),
                            Err(e) => {
                                log::warn!("transcription failed: {e:#}");
                                String::new()
                            }
                        };
                        if text.trim().is_empty() {
                            events.caption(CaptionEvent {
                                kind: "clear",
                                source: j.source,
                                id: None,
                                start_ms: j.start_ms,
                                end_ms: j.end_ms,
                                speaker: j.speaker,
                                text: String::new(),
                            });
                            continue;
                        }
                        session.on_final(Segment {
                            id: j.id,
                            source: j.source,
                            start_ms: j.start_ms,
                            end_ms: j.end_ms,
                            speaker: j.speaker,
                            text,
                            cluster: j.cluster,
                        });
                    }
                    Job::Partial(j) => {
                        let opts = DecodeOptions { language: &language, initial_prompt: &prompt, partial: true };
                        if let Ok(segs) = engine.transcribe(&j.samples, &opts) {
                            let text = segs
                                .iter()
                                .filter_map(|s| clean_segment_text(&s.text, s.no_speech))
                                .collect::<Vec<_>>()
                                .join(" ");
                            if !text.is_empty() {
                                events.caption(CaptionEvent {
                                    kind: "partial",
                                    source: j.source,
                                    id: None,
                                    start_ms: j.start_ms,
                                    end_ms: j.end_ms,
                                    speaker: j.speaker_hint,
                                    text,
                                });
                            }
                        }
                    }
                }
            }

            if settings.keep_model_loaded {
                *slot.lock() = Some(engine);
                events.engine("idle", "Whisper model kept in memory", None);
            } else {
                drop(engine); // frees VRAM
                events.engine("idle", "Whisper model unloaded (memory freed)", None);
            }
        })
        .expect("spawn worker")
}
