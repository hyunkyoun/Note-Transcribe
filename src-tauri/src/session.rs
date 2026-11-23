//! An active recording: capture threads, pipelines, the Whisper worker, and
//! incremental (crash-safe) transcript files.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Local};
use meeting_core::echo::{is_echo, remove_mic_echoes};
use meeting_core::labels::{label_for, Renames};
use meeting_core::merge::{render_line, sort_segments};
use meeting_core::settings::Settings;
use meeting_core::{Segment, Source};
use parking_lot::Mutex;

use crate::audio::{self, CaptureHandle};
use crate::diarize::SharedExtractor;
use crate::events::{CaptionEvent, Events};
use crate::meeting::Meeting;
use crate::pipeline::{spawn_pipeline, PipelineConfig, PipelineResult};
use crate::transcriber::EngineInfo;
use crate::vad::Vad;
use crate::worker::{spawn_worker, EngineSlot, JobQueue};

/// Appends every final segment to `<id>.jsonl` and `<id>.txt` as it arrives.
struct LiveWriter {
    jsonl: File,
    txt: File,
}

impl LiveWriter {
    fn open(dir: &std::path::Path, id: &str) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let open = |ext: &str| OpenOptions::new().create(true).append(true).open(dir.join(format!("{id}.{ext}")));
        Ok(Self { jsonl: open("jsonl")?, txt: open("txt")? })
    }

    fn append(&mut self, seg: &Segment) {
        if let Ok(line) = serde_json::to_string(seg) {
            let _ = writeln!(self.jsonl, "{line}");
        }
        let _ = writeln!(self.txt, "{}", render_line(seg, &Renames::default()));
        let _ = self.jsonl.flush();
        let _ = self.txt.flush();
        // Push to disk so a crash (even of the OS) loses at most the current line.
        let _ = self.jsonl.sync_data();
    }
}

/// State shared between the worker thread and the session owner.
pub struct SessionShared {
    pub id: String,
    pub dir: PathBuf,
    pub started: DateTime<Local>,
    pub t0: Instant,
    settings: Settings,
    segments: Mutex<Vec<Segment>>,
    writer: Mutex<LiveWriter>,
    engine_info: Mutex<Option<EngineInfo>>,
    events: Events,
}

impl SessionShared {
    pub fn set_engine_info(&self, info: EngineInfo) {
        *self.engine_info.lock() = Some(info);
    }

    pub fn on_final(&self, seg: Segment) {
        // Live echo check: drop mic text that duplicates recent desktop text.
        if self.settings.suppress_mic_echo && seg.source == Source::Mic {
            let segs = self.segments.lock();
            if segs.iter().rev().take(20).any(|d| is_echo(&seg, d)) {
                log::info!("dropped mic echo segment {}", seg.id);
                self.events.caption(CaptionEvent {
                    kind: "clear",
                    source: seg.source,
                    id: None,
                    start_ms: seg.start_ms,
                    end_ms: seg.end_ms,
                    speaker: seg.speaker,
                    text: String::new(),
                });
                return;
            }
        }
        self.writer.lock().append(&seg);
        self.events.caption(CaptionEvent {
            kind: "final",
            source: seg.source,
            id: Some(seg.id),
            start_ms: seg.start_ms,
            end_ms: seg.end_ms,
            speaker: seg.speaker.clone(),
            text: seg.text.clone(),
        });
        self.segments.lock().push(seg);
    }
}

pub struct ActiveRecording {
    pub shared: Arc<SessionShared>,
    captures: Vec<CaptureHandle>,
    pipelines: Vec<JoinHandle<PipelineResult>>,
    queue: Arc<JobQueue>,
    worker: Option<JoinHandle<()>>,
}

fn unique_id(dir: &std::path::Path, started: &DateTime<Local>) -> String {
    let base = started.format("%Y-%m-%d_%H%M").to_string();
    if !dir.join(format!("{base}.jsonl")).exists() {
        return base;
    }
    (2..).map(|n| format!("{base}_{n}")).find(|id| !dir.join(format!("{id}.jsonl")).exists()).unwrap_or(base)
}

impl ActiveRecording {
    pub fn start(
        settings: Settings,
        models_dir: PathBuf,
        transcripts_dir: PathBuf,
        engine_slot: EngineSlot,
        extractor: Option<SharedExtractor>,
        gpu_expected: bool,
        events: Events,
    ) -> Result<Self> {
        if !settings.capture_desktop && !settings.capture_mic {
            bail!("Both sources are turned off. Enable desktop audio and/or the microphone.");
        }
        let started = Local::now();
        let id = unique_id(&transcripts_dir, &started);
        let writer = LiveWriter::open(&transcripts_dir, &id)
            .with_context(|| format!("cannot write transcripts to {}", transcripts_dir.display()))?;
        let t0 = Instant::now();
        let shared = Arc::new(SessionShared {
            id: id.clone(),
            dir: transcripts_dir.clone(),
            started,
            t0,
            settings: settings.clone(),
            segments: Mutex::new(Vec::new()),
            writer: Mutex::new(writer),
            engine_info: Mutex::new(None),
            events: events.clone(),
        });

        let queue = Arc::new(JobQueue::default());
        let ids = Arc::new(AtomicU64::new(1));
        let worker = spawn_worker(queue.clone(), engine_slot, models_dir.clone(), settings.clone(), shared.clone(), events.clone());

        let mut captures = Vec::new();
        let mut pipelines = Vec::new();
        let sources = [
            (Source::Desktop, settings.capture_desktop, settings.output_device.clone()),
            (Source::Mic, settings.capture_mic, settings.mic_device.clone()),
        ];
        for (source, enabled, device) in sources {
            if !enabled {
                continue;
            }
            let (vad, warning) = Vad::new(&models_dir);
            if let Some(w) = warning {
                events.status(Some(source), audio::Level::Warn, w);
            }
            let (tx, rx) = crossbeam_channel::unbounded();
            let cfg = PipelineConfig {
                source,
                settings: settings.clone(),
                wav_path: settings.save_raw_audio.then(|| transcripts_dir.join(format!("{id}.{}.wav", source.as_str()))),
                vad,
                extractor: extractor.clone(),
                partial_interval: if gpu_expected { Duration::from_millis(1_000) } else { Duration::from_millis(2_500) },
            };
            pipelines.push(spawn_pipeline(cfg, rx, queue.clone(), ids.clone(), t0, events.clone()));
            captures.push(audio::spawn_capture(source, device, t0, tx));
        }
        log::info!("recording {id} started");
        Ok(Self { shared, captures, pipelines, queue, worker: Some(worker) })
    }

    pub fn shared_engine_info(&self) -> Option<EngineInfo> {
        self.shared.engine_info.lock().clone()
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.shared.t0.elapsed().as_millis() as u64
    }

    pub fn pending_jobs(&self) -> usize {
        self.queue.pending_finals()
    }

    /// Stop capture, finish transcribing what's queued (unless `fast`), refine
    /// speaker labels, remove mic echoes and write the final files.
    pub fn stop(mut self, fast: bool) -> Meeting {
        let duration_ms = self.elapsed_ms();
        for c in self.captures.drain(..) {
            c.stop(); // drops the sender -> pipeline flushes VAD and exits
        }
        let results: Vec<PipelineResult> = self.pipelines.drain(..).filter_map(|p| p.join().ok()).collect();
        if fast {
            // App is closing: give the GPU a few seconds to finish queued utterances.
            let deadline = Instant::now() + Duration::from_secs(5);
            while self.queue.pending_finals() > 0 && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(100));
            }
        }
        self.queue.close(fast);
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }

        let s = &self.shared.settings;
        let mut segments = std::mem::take(&mut *self.shared.segments.lock());

        // Re-cluster speakers with the whole meeting's evidence.
        for r in &results {
            if let Some(d) = &r.diarizer {
                let map = d.refined();
                for seg in segments.iter_mut().filter(|x| x.source == r.source) {
                    if let Some(&c) = map.get(&seg.id) {
                        seg.cluster = Some(c);
                        seg.speaker = label_for(seg.source, s.mic_is_only_me, &s.my_name, Some(c));
                    }
                }
            }
        }
        if s.suppress_mic_echo {
            let (kept, removed) = remove_mic_echoes(&segments);
            if !removed.is_empty() {
                log::info!("removed {} mic echo segments", removed.len());
            }
            segments = kept;
        }
        sort_segments(&mut segments);

        let meeting = Meeting {
            id: self.shared.id.clone(),
            started_label: self.shared.started.format("%Y-%m-%d %H:%M").to_string(),
            duration_ms,
            dir: self.shared.dir.clone(),
            segments,
            renames: Renames::default(),
            summary: None,
            model: self.shared.engine_info.lock().as_ref().map(|i| i.model.clone()),
            discord_posted: false,
        };
        if let Err(e) = meeting.save() {
            log::error!("saving final transcript failed: {e:#} (the .jsonl crash log is still on disk)");
        }
        log::info!("recording {} finished: {} segments", meeting.id, meeting.segments.len());
        meeting
    }
}
