//! Per-stream processing thread:
//! raw device audio -> mono -> 16 kHz -> wall-clock alignment -> (optional WAV)
//! -> VAD -> diarization -> transcription jobs (final + live partials).

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError};
use meeting_core::labels::{label_for, should_diarize};
use meeting_core::resample::{downmix, StreamResampler};
use meeting_core::settings::Settings;
use meeting_core::timeline::{samples_to_ms, Adjust, Timeline};
use meeting_core::Source;

use crate::audio::{CaptureEvent, Level};
use crate::diarize::{Diarizer, SharedExtractor};
use crate::events::Events;
use crate::vad::{Vad, SAMPLE_RATE};
use crate::worker::{FinalJob, JobQueue, PartialJob};

const RING_SECONDS: usize = 30;
const PRE_ROLL: u64 = SAMPLE_RATE as u64 * 3 / 10; // 300 ms
const MIN_PARTIAL: u64 = SAMPLE_RATE as u64 * 8 / 10; // 0.8 s
const MAX_PARTIAL: u64 = SAMPLE_RATE as u64 * 12; // only preview the last 12 s
const MIN_FINAL: usize = SAMPLE_RATE as usize * 3 / 10; // ignore < 0.3 s blips
/// After this long without packets, synthesize silence up to "now".
const IDLE_AFTER: Duration = Duration::from_millis(300);
const FEED_CHUNK: usize = SAMPLE_RATE as usize; // feed synthetic silence 1 s at a time

pub struct PipelineConfig {
    pub source: Source,
    pub settings: Settings,
    pub wav_path: Option<PathBuf>,
    pub vad: Vad,
    pub extractor: Option<SharedExtractor>,
    /// How often to refresh the live partial caption while someone is talking.
    pub partial_interval: Duration,
}

/// What a pipeline hands back when its stream ends.
pub struct PipelineResult {
    pub source: Source,
    pub diarizer: Option<Diarizer>,
}

struct Ring {
    buf: VecDeque<f32>,
    start: u64,
}

impl Ring {
    fn push(&mut self, samples: &[f32]) {
        self.buf.extend(samples.iter().copied());
        let cap = RING_SECONDS * SAMPLE_RATE as usize;
        if self.buf.len() > cap {
            let excess = self.buf.len() - cap;
            self.buf.drain(..excess);
            self.start += excess as u64;
        }
    }
    fn end(&self) -> u64 {
        self.start + self.buf.len() as u64
    }
    fn range(&self, from: u64, to: u64) -> Vec<f32> {
        let from = from.max(self.start);
        let to = to.min(self.end());
        if from >= to {
            return Vec::new();
        }
        self.buf.range((from - self.start) as usize..(to - self.start) as usize).copied().collect()
    }
}

struct Pipeline {
    source: Source,
    settings: Settings,
    vad: Vad,
    diarizer: Option<Diarizer>,
    ring: Ring,
    wav: Option<hound::WavWriter<std::io::BufWriter<std::fs::File>>>,
    queue: Arc<JobQueue>,
    ids: Arc<AtomicU64>,
    partial_interval: Duration,
    speaking: bool,
    onset: u64,
    last_partial: Instant,
    last_final_end: u64,
    last_label: String,
}

impl Pipeline {
    /// Push aligned 16 kHz audio through WAV / ring / VAD and emit jobs.
    fn feed(&mut self, audio: &[f32]) {
        if audio.is_empty() {
            return;
        }
        if let Some(w) = self.wav.as_mut() {
            for &x in audio {
                let _ = w.write_sample((x.clamp(-1.0, 1.0) * i16::MAX as f32) as i16);
            }
        }
        self.ring.push(audio);
        self.vad.accept(audio);
        self.take_segments();
        self.maybe_partial();
    }

    fn take_segments(&mut self) {
        let s = &self.settings;
        while let Some((start, samples)) = self.vad.pop() {
            if samples.len() < MIN_FINAL {
                continue;
            }
            let id = self.ids.fetch_add(1, Ordering::SeqCst);
            let cluster = self.diarizer.as_mut().map(|d| d.assign(id, &samples));
            let speaker = label_for(self.source, s.mic_is_only_me, &s.my_name, cluster);
            self.last_label = speaker.clone();
            let end = start + samples.len() as u64;
            self.last_final_end = self.last_final_end.max(end);
            self.queue.push_final(FinalJob {
                source: self.source,
                id,
                start_ms: samples_to_ms(SAMPLE_RATE, start),
                end_ms: samples_to_ms(SAMPLE_RATE, end),
                samples,
                speaker,
                cluster,
            });
        }
    }

    /// Live partial captions while speech is ongoing.
    fn maybe_partial(&mut self) {
        let now_speaking = self.vad.is_speech();
        let pos = self.ring.end();
        if now_speaking && !self.speaking {
            // Detection lags the real onset by ~min_speech_duration; back up a bit.
            self.onset = pos.saturating_sub(PRE_ROLL + SAMPLE_RATE as u64 / 4).max(self.last_final_end);
            self.last_partial = Instant::now();
        }
        self.speaking = now_speaking;
        if !self.speaking {
            return;
        }
        let from = self.onset.max(self.last_final_end).max(pos.saturating_sub(MAX_PARTIAL));
        if pos.saturating_sub(from) >= MIN_PARTIAL && self.last_partial.elapsed() >= self.partial_interval {
            self.last_partial = Instant::now();
            self.queue.push_partial(PartialJob {
                source: self.source,
                start_ms: samples_to_ms(SAMPLE_RATE, from),
                end_ms: samples_to_ms(SAMPLE_RATE, pos),
                samples: self.ring.range(from, pos),
                speaker_hint: self.last_label.clone(),
            });
        }
    }

    fn feed_silence(&mut self, mut n: usize) {
        while n > 0 {
            let k = n.min(FEED_CHUNK);
            self.feed(&vec![0.0; k]);
            n -= k;
        }
    }
}

pub fn spawn_pipeline(
    cfg: PipelineConfig,
    rx: Receiver<CaptureEvent>,
    queue: Arc<JobQueue>,
    ids: Arc<AtomicU64>,
    t0: Instant,
    events: Events,
) -> JoinHandle<PipelineResult> {
    std::thread::Builder::new()
        .name(format!("pipeline-{}", cfg.source.as_str()))
        .spawn(move || {
            let source = cfg.source;
            let s = cfg.settings.clone();
            let wav = cfg.wav_path.as_ref().and_then(|p| {
                let spec = hound::WavSpec { channels: 1, sample_rate: SAMPLE_RATE, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
                hound::WavWriter::create(p, spec)
                    .map_err(|e| events.status(Some(source), Level::Warn, format!("cannot save raw audio: {e}")))
                    .ok()
            });
            let mut p = Pipeline {
                source,
                diarizer: should_diarize(source, s.mic_is_only_me)
                    .then(|| Diarizer::new(cfg.extractor.clone(), s.diarization_threshold, s.max_speakers)),
                last_label: label_for(source, s.mic_is_only_me, &s.my_name, None),
                settings: s,
                vad: cfg.vad,
                ring: Ring { buf: VecDeque::new(), start: 0 },
                wav,
                queue,
                ids,
                partial_interval: cfg.partial_interval,
                speaking: false,
                onset: 0,
                last_partial: Instant::now(),
                last_final_end: 0,
            };
            let mut resampler: Option<StreamResampler> = None;
            let mut timeline = Timeline::new(SAMPLE_RATE, 250, 750);
            let mut last_packet = Instant::now();

            loop {
                let ev = match rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(ev) => ev,
                    Err(RecvTimeoutError::Timeout) => {
                        // Loopback delivers nothing while the output is silent: keep the
                        // clock moving so the VAD can close the last utterance promptly.
                        if last_packet.elapsed() >= IDLE_AFTER {
                            let n = timeline.idle_fill(t0.elapsed().as_secs_f64(), 100);
                            p.feed_silence(n);
                        }
                        continue;
                    }
                    Err(RecvTimeoutError::Disconnected) => break, // capture stopped
                };
                let packet = match ev {
                    CaptureEvent::Audio(pk) => pk,
                    CaptureEvent::Status { level, message } => {
                        events.status(Some(source), level, message);
                        continue;
                    }
                };
                last_packet = Instant::now();

                let mono = downmix(&packet.samples, packet.channels as usize);
                if resampler.as_ref().map(|r| r.in_rate()) != Some(packet.sample_rate) {
                    // First packet, or the device was re-opened with a different rate.
                    resampler = Some(StreamResampler::new(packet.sample_rate, SAMPLE_RATE));
                }
                let mut audio = resampler.as_mut().map(|r| r.process(&mono)).unwrap_or_default();
                match timeline.on_batch(audio.len(), packet.elapsed) {
                    Adjust::None => {}
                    Adjust::Pad(n) => p.feed_silence(n),
                    Adjust::Drop(n) => {
                        audio.drain(..n.min(audio.len()));
                    }
                }
                p.feed(&audio);
            }

            // Stream ended: close any open speech segment.
            p.vad.flush();
            p.take_segments();
            if let Some(w) = p.wav.take() {
                let _ = w.finalize();
            }
            PipelineResult { source, diarizer: p.diarizer }
        })
        .expect("spawn pipeline")
}
