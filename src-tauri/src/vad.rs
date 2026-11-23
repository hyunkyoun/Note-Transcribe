//! Voice activity detection: Silero VAD (sherpa-onnx) when `silero_vad.onnx`
//! is installed, otherwise the energy-based fallback from `meeting-core`.

use std::path::Path;

use meeting_core::vad::EnergyVad;
use sherpa_rs::silero_vad::{SileroVad, SileroVadConfig};

pub const SAMPLE_RATE: u32 = 16_000;
const WINDOW: usize = 512;

pub enum Vad {
    Silero { vad: SileroVad, pending: Vec<f32> },
    Energy(EnergyVad),
}

impl Vad {
    /// Returns the detector and, if Silero couldn't be used, a warning for the UI.
    pub fn new(models_dir: &Path) -> (Vad, Option<String>) {
        let path = models_dir.join(crate::paths::VAD_MODEL_FILE);
        if path.is_file() {
            let config = SileroVadConfig {
                model: path.to_string_lossy().to_string(),
                threshold: 0.5,
                min_silence_duration: 0.45,
                min_speech_duration: 0.25,
                // Long monologues are split so a final caption arrives at least this often.
                max_speech_duration: 12.0,
                sample_rate: SAMPLE_RATE,
                window_size: WINDOW as i32,
                provider: None,
                num_threads: Some(1),
                debug: false,
            };
            match SileroVad::new(config, 60.0) {
                Ok(vad) => return (Vad::Silero { vad, pending: Vec::with_capacity(WINDOW * 4) }, None),
                Err(e) => {
                    log::warn!("Silero VAD failed to load: {e}");
                    return (
                        Vad::Energy(EnergyVad::new(SAMPLE_RATE)),
                        Some(format!("Silero VAD failed to load ({e}); using basic energy VAD")),
                    );
                }
            }
        }
        (
            Vad::Energy(EnergyVad::new(SAMPLE_RATE)),
            Some(format!(
                "{} not found in {}; using basic energy VAD (more false triggers on noise)",
                crate::paths::VAD_MODEL_FILE,
                models_dir.display()
            )),
        )
    }

    pub fn accept(&mut self, samples: &[f32]) {
        match self {
            Vad::Silero { vad, pending } => {
                pending.extend_from_slice(samples);
                let mut off = 0;
                while pending.len() - off >= WINDOW {
                    vad.accept_waveform(pending[off..off + WINDOW].to_vec());
                    off += WINDOW;
                }
                pending.drain(..off);
            }
            Vad::Energy(v) => v.accept(samples),
        }
    }

    pub fn is_speech(&mut self) -> bool {
        match self {
            Vad::Silero { vad, .. } => vad.is_speech(),
            Vad::Energy(v) => v.is_speech(),
        }
    }

    /// Next finished speech segment: (start sample since stream start, samples).
    pub fn pop(&mut self) -> Option<(u64, Vec<f32>)> {
        match self {
            Vad::Silero { vad, .. } => {
                if vad.is_empty() {
                    None
                } else {
                    let seg = vad.front();
                    vad.pop();
                    Some((seg.start.max(0) as u64, seg.samples))
                }
            }
            Vad::Energy(v) => v.pop(),
        }
    }

    pub fn flush(&mut self) {
        match self {
            Vad::Silero { vad, pending } => {
                if !pending.is_empty() {
                    let mut last = std::mem::take(pending);
                    last.resize(WINDOW, 0.0);
                    vad.accept_waveform(last);
                }
                vad.flush();
            }
            Vad::Energy(v) => v.flush(),
        }
    }
}
