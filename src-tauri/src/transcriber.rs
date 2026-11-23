//! Local speech-to-text with whisper.cpp (whisper-rs), CUDA when available.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use meeting_core::settings::Settings;
use serde::Serialize;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};

use crate::paths;
use crate::vad::SAMPLE_RATE;

#[derive(Debug, Clone, Serialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_total_mb: u64,
    pub vram_free_mb: u64,
}

/// GPUs visible to ggml (empty when built without CUDA or no NVIDIA GPU/driver).
pub fn gpu_devices() -> Vec<GpuInfo> {
    use whisper_rs::whisper_rs_sys as sys;
    let mut out = Vec::new();
    // SAFETY: plain queries against ggml's static backend registry.
    unsafe {
        let n = sys::ggml_backend_dev_count();
        for i in 0..n {
            let dev = sys::ggml_backend_dev_get(i);
            if dev.is_null() || sys::ggml_backend_dev_type(dev) != sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU {
                continue;
            }
            let desc = sys::ggml_backend_dev_description(dev);
            let name = if desc.is_null() {
                "GPU".to_string()
            } else {
                std::ffi::CStr::from_ptr(desc).to_string_lossy().to_string()
            };
            let (mut free, mut total) = (0usize, 0usize);
            sys::ggml_backend_dev_memory(dev, &mut free, &mut total);
            out.push(GpuInfo { name, vram_total_mb: (total / 1_048_576) as u64, vram_free_mb: (free / 1_048_576) as u64 });
        }
    }
    out
}

#[derive(Debug, Clone, Serialize)]
pub struct EngineInfo {
    pub model: String,
    pub path: String,
    pub gpu: bool,
    pub gpu_name: Option<String>,
    /// Approximate VRAM used by the model (free memory before vs after load).
    pub vram_used_mb: Option<u64>,
    pub note: Option<String>,
}

pub struct RawSegment {
    pub text: String,
    pub no_speech: f32,
}

pub struct DecodeOptions<'a> {
    pub language: &'a str,
    pub initial_prompt: &'a str,
    /// Quick, lower-quality pass for live partial captions.
    pub partial: bool,
}

pub struct WhisperEngine {
    // `state` keeps its own reference to the context; keep both alive together.
    state: WhisperState,
    _ctx: WhisperContext,
    pub info: EngineInfo,
    threads: i32,
}

fn is_large(model: &str) -> bool {
    let m = model.to_lowercase();
    m.contains("large") || m.contains("medium")
}

/// Resolve which model file to use, without loading it. Used as a fast
/// preflight check when Start is pressed.
pub fn resolve_model(models_dir: &Path, settings: &Settings) -> Result<(String, PathBuf)> {
    let candidates = [settings.whisper_model.as_str(), settings.fallback_model.as_str()];
    for m in candidates {
        let p = models_dir.join(paths::whisper_file_name(m));
        if p.is_file() {
            return Ok((m.to_string(), p));
        }
    }
    // Anything at all in the folder, smallest name order is fine as a last resort.
    if let Some(m) = paths::list_whisper_models(models_dir).into_iter().next() {
        let p = models_dir.join(paths::whisper_file_name(&m));
        return Ok((m, p));
    }
    Err(anyhow!(
        "No Whisper model found. Expected {} in {} (run scripts\\download-models.ps1 or set the models folder in Settings)",
        paths::whisper_file_name(&settings.whisper_model),
        models_dir.display()
    ))
}

impl WhisperEngine {
    pub fn load(models_dir: &Path, settings: &Settings) -> Result<Self> {
        whisper_rs::install_logging_hooks();
        let gpu_wanted = settings.use_gpu && cfg!(feature = "cuda");
        let gpus = if gpu_wanted { gpu_devices() } else { Vec::new() };
        let mut note = None;

        let (mut model, mut path) = resolve_model(models_dir, settings)?;
        let use_gpu = !gpus.is_empty();
        if gpu_wanted && !use_gpu {
            // CUDA requested but unavailable: a large model on CPU can't keep up live.
            let fb = models_dir.join(paths::whisper_file_name(&settings.fallback_model));
            if is_large(&model) && fb.is_file() {
                note = Some(format!(
                    "CUDA GPU not available; using CPU with the fallback model \"{}\"",
                    settings.fallback_model
                ));
                model = settings.fallback_model.clone();
                path = fb;
            } else {
                note = Some("CUDA GPU not available; running Whisper on CPU".into());
            }
        } else if !settings.use_gpu {
            note = Some("GPU disabled in Settings; running Whisper on CPU".into());
        } else if !cfg!(feature = "cuda") {
            note = Some("This build has no CUDA support; running Whisper on CPU".into());
        }
        if model != settings.whisper_model && note.is_none() {
            note = Some(format!(
                "{} not found; using \"{model}\" instead",
                paths::whisper_file_name(&settings.whisper_model)
            ));
        }

        let free_before = gpus.first().map(|g| g.vram_free_mb);
        let try_load = |gpu: bool| -> Result<WhisperContext> {
            let mut params = WhisperContextParameters::default();
            params.use_gpu(gpu).flash_attn(gpu).gpu_device(0);
            WhisperContext::new_with_params(&path, params)
                .map_err(|e| anyhow!("{e:?}"))
                .with_context(|| format!("loading {}", path.display()))
        };
        let (ctx, gpu) = match try_load(use_gpu) {
            Ok(c) => (c, use_gpu),
            Err(e) if use_gpu => {
                log::warn!("GPU load failed ({e:#}); retrying on CPU");
                note = Some(format!("GPU load failed; running on CPU ({e:#})"));
                (try_load(false)?, false)
            }
            Err(e) => return Err(e),
        };
        let state = ctx.create_state().map_err(|e| anyhow!("create whisper state: {e:?}"))?;
        let vram_used_mb = if gpu {
            let after = gpu_devices().first().map(|g| g.vram_free_mb);
            free_before.zip(after).map(|(b, a)| b.saturating_sub(a))
        } else {
            None
        };
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) as i32;
        let threads = if gpu { 4.min(cores) } else { (cores - 2).clamp(1, 8) };
        let info = EngineInfo {
            model: model.clone(),
            path: path.display().to_string(),
            gpu,
            gpu_name: if gpu { gpus.first().map(|g| g.name.clone()) } else { None },
            vram_used_mb,
            note,
        };
        log::info!("whisper loaded: {info:?}");
        Ok(Self { state, _ctx: ctx, info, threads })
    }

    pub fn transcribe(&mut self, samples: &[f32], opts: &DecodeOptions) -> Result<Vec<RawSegment>> {
        // whisper.cpp ignores input shorter than ~1 s; pad with silence.
        let min = SAMPLE_RATE as usize * 105 / 100;
        let padded;
        let audio = if samples.len() < min {
            padded = {
                let mut v = samples.to_vec();
                v.resize(min, 0.0);
                v
            };
            &padded[..]
        } else {
            samples
        };

        let strategy = if opts.partial || !self.info.gpu {
            SamplingStrategy::Greedy { best_of: 1 }
        } else {
            SamplingStrategy::BeamSearch { beam_size: 3, patience: -1.0 }
        };
        let mut p = FullParams::new(strategy);
        p.set_n_threads(self.threads);
        p.set_language(Some(opts.language));
        p.set_translate(false);
        p.set_no_context(true); // don't carry text between segments (reduces hallucination loops)
        p.set_single_segment(opts.partial);
        p.set_no_timestamps(opts.partial);
        p.set_print_special(false);
        p.set_print_progress(false);
        p.set_print_realtime(false);
        p.set_print_timestamps(false);
        p.set_suppress_blank(true);
        p.set_suppress_nst(true);
        p.set_temperature(0.0);
        p.set_temperature_inc(if opts.partial { 0.0 } else { 0.2 });
        p.set_no_speech_thold(0.6);
        if !opts.initial_prompt.trim().is_empty() {
            p.set_initial_prompt(opts.initial_prompt.trim());
        }

        self.state.full(p, audio).map_err(|e| anyhow!("whisper inference failed: {e:?}"))?;
        let mut out = Vec::new();
        for i in 0..self.state.full_n_segments() {
            if let Some(seg) = self.state.get_segment(i) {
                let text = seg.to_str_lossy().map(|c| c.to_string()).unwrap_or_default();
                out.push(RawSegment {
                    text,
                    no_speech: seg.no_speech_probability(),
                });
            }
        }
        Ok(out)
    }
}
