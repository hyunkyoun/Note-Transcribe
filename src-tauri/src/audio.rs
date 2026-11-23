//! Audio capture via cpal's WASAPI backend.
//!
//! * Desktop audio: an *input* stream opened on a *render* (output) device.
//!   cpal sets `AUDCLNT_STREAMFLAGS_LOOPBACK` for this, which is passive
//!   loopback capture of whatever that device plays. No virtual cable needed.
//! * Microphone: a normal capture stream.
//!
//! Each capture runs on its own thread that owns the cpal stream (streams are
//! not `Send` on every platform) and re-opens the device if it disappears
//! (headphones unplugged) or, when following the Windows default, if the
//! default device changes. It never panics on device errors.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::Sender;
use meeting_core::Source;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct DeviceInfo {
    pub name: String,
    pub is_default: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct DeviceList {
    pub outputs: Vec<DeviceInfo>,
    pub inputs: Vec<DeviceInfo>,
}

fn list(kind: Source) -> Vec<DeviceInfo> {
    let host = cpal::default_host();
    let default = match kind {
        Source::Desktop => host.default_output_device(),
        Source::Mic => host.default_input_device(),
    }
    .and_then(|d| d.name().ok());
    let devices = match kind {
        Source::Desktop => host.output_devices(),
        Source::Mic => host.input_devices(),
    };
    let mut out: Vec<DeviceInfo> = match devices {
        Ok(it) => it
            .filter_map(|d| d.name().ok())
            .map(|name| DeviceInfo { is_default: Some(&name) == default.as_ref(), name })
            .collect(),
        Err(e) => {
            log::warn!("could not enumerate {} devices: {e}", kind.as_str());
            Vec::new()
        }
    };
    out.dedup_by(|a, b| a.name == b.name);
    out
}

pub fn list_devices() -> DeviceList {
    DeviceList { outputs: list(Source::Desktop), inputs: list(Source::Mic) }
}

/// Raw interleaved audio straight from the device callback.
pub struct AudioPacket {
    pub samples: Vec<f32>,
    pub channels: u16,
    pub sample_rate: u32,
    /// Seconds since the recording started, measured when the callback ran.
    pub elapsed: f64,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Info,
    Warn,
    Error,
}

pub enum CaptureEvent {
    Audio(AudioPacket),
    Status { level: Level, message: String },
}

pub struct CaptureHandle {
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl CaptureHandle {
    /// Stop capturing and wait for the device to be released.
    pub fn stop(mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

impl Drop for CaptureHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
    }
}

fn find_device(kind: Source, wanted: Option<&str>) -> Option<(cpal::Device, String, bool)> {
    let host = cpal::default_host();
    if let Some(w) = wanted {
        let devices = match kind {
            Source::Desktop => host.output_devices().ok(),
            Source::Mic => host.input_devices().ok(),
        };
        if let Some(d) = devices.into_iter().flatten().find(|d| d.name().map(|n| n == w).unwrap_or(false)) {
            return Some((d, w.to_string(), false));
        }
    }
    let d = match kind {
        Source::Desktop => host.default_output_device(),
        Source::Mic => host.default_input_device(),
    }?;
    let name = d.name().unwrap_or_else(|_| "default device".into());
    Some((d, name, wanted.is_some()))
}

fn default_name(kind: Source) -> Option<String> {
    let host = cpal::default_host();
    match kind {
        Source::Desktop => host.default_output_device(),
        Source::Mic => host.default_input_device(),
    }
    .and_then(|d| d.name().ok())
}

fn device_exists(kind: Source, name: &str) -> bool {
    let host = cpal::default_host();
    let devices = match kind {
        Source::Desktop => host.output_devices().ok(),
        Source::Mic => host.input_devices().ok(),
    };
    devices.into_iter().flatten().any(|d| d.name().map(|n| n == name).unwrap_or(false))
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    t0: Instant,
    tx: Sender<CaptureEvent>,
    failed: Arc<AtomicBool>,
) -> Result<cpal::Stream>
where
    T: cpal::SizedSample + Send + 'static,
    f32: cpal::FromSample<T>,
{
    let channels = config.channels;
    let sample_rate = config.sample_rate.0;
    let stream = device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            let samples: Vec<f32> = data.iter().map(|s| <f32 as cpal::FromSample<T>>::from_sample_(*s)).collect();
            let _ = tx.send(CaptureEvent::Audio(AudioPacket {
                samples,
                channels,
                sample_rate,
                elapsed: t0.elapsed().as_secs_f64(),
            }));
        },
        move |err| {
            log::warn!("audio stream error: {err}");
            failed.store(true, Ordering::SeqCst);
        },
        None,
    )?;
    Ok(stream)
}

fn open(kind: Source, device: &cpal::Device, t0: Instant, tx: Sender<CaptureEvent>, failed: Arc<AtomicBool>) -> Result<(cpal::Stream, String)> {
    let supported = match kind {
        // Loopback uses the render device's mix format.
        Source::Desktop => device.default_output_config(),
        Source::Mic => device.default_input_config(),
    }
    .context("query device format")?;
    let format = supported.sample_format();
    let config: cpal::StreamConfig = supported.into();
    let desc = format!("{} ch @ {} Hz {:?}", config.channels, config.sample_rate.0, format);
    let stream = match format {
        cpal::SampleFormat::F32 => build_stream::<f32>(device, &config, t0, tx, failed),
        cpal::SampleFormat::I16 => build_stream::<i16>(device, &config, t0, tx, failed),
        cpal::SampleFormat::U16 => build_stream::<u16>(device, &config, t0, tx, failed),
        cpal::SampleFormat::I32 => build_stream::<i32>(device, &config, t0, tx, failed),
        cpal::SampleFormat::F64 => build_stream::<f64>(device, &config, t0, tx, failed),
        other => Err(anyhow!("unsupported sample format {other:?}")),
    }
    .context("open stream")?;
    stream.play().context("start stream")?;
    Ok((stream, desc))
}

/// Start capturing `kind` from `wanted` (None = follow the Windows default).
pub fn spawn_capture(kind: Source, wanted: Option<String>, t0: Instant, tx: Sender<CaptureEvent>) -> CaptureHandle {
    let stop = Arc::new(AtomicBool::new(false));
    let stop2 = stop.clone();
    let join = std::thread::Builder::new()
        .name(format!("capture-{}", kind.as_str()))
        .spawn(move || capture_loop(kind, wanted, t0, tx, stop2))
        .expect("spawn capture thread");
    CaptureHandle { stop, join: Some(join) }
}

fn capture_loop(kind: Source, wanted: Option<String>, t0: Instant, tx: Sender<CaptureEvent>, stop: Arc<AtomicBool>) {
    let label = match kind {
        Source::Desktop => "Desktop audio",
        Source::Mic => "Microphone",
    };
    let status = |level, message: String| {
        log::info!("{label}: {message}");
        let _ = tx.send(CaptureEvent::Status { level, message });
    };
    let mut last_error = String::new();

    while !stop.load(Ordering::SeqCst) {
        let Some((device, name, is_fallback)) = find_device(kind, wanted.as_deref()) else {
            let msg = "no device available; waiting for one to appear".to_string();
            if msg != last_error {
                status(Level::Error, msg.clone());
                last_error = msg;
            }
            std::thread::sleep(Duration::from_secs(2));
            continue;
        };
        let failed = Arc::new(AtomicBool::new(false));
        match open(kind, &device, t0, tx.clone(), failed.clone()) {
            Ok((stream, desc)) => {
                last_error.clear();
                if is_fallback {
                    status(
                        Level::Warn,
                        format!("\"{}\" not found, using default \"{name}\" ({desc})", wanted.as_deref().unwrap_or_default()),
                    );
                } else {
                    status(Level::Info, format!("capturing \"{name}\" ({desc})"));
                }
                let mut ticks = 0u32;
                loop {
                    std::thread::sleep(Duration::from_millis(100));
                    if stop.load(Ordering::SeqCst) {
                        break;
                    }
                    if failed.load(Ordering::SeqCst) {
                        status(Level::Warn, format!("\"{name}\" stopped (unplugged or changed); reconnecting…"));
                        break;
                    }
                    ticks += 1;
                    if ticks.is_multiple_of(20) {
                        // Every ~2 s: follow default-device changes / return to the preferred device.
                        if wanted.is_none() && default_name(kind).is_some_and(|d| d != name) {
                            status(Level::Info, "Windows default device changed; switching".into());
                            break;
                        }
                        if is_fallback && wanted.as_deref().is_some_and(|w| device_exists(kind, w)) {
                            status(Level::Info, "preferred device is back; switching".into());
                            break;
                        }
                    }
                }
                drop(stream);
            }
            Err(e) => {
                let msg = format!("cannot open \"{name}\": {e:#}; retrying");
                if msg != last_error {
                    status(Level::Error, msg.clone());
                    last_error = msg;
                }
                std::thread::sleep(Duration::from_secs(2));
            }
        }
    }
}
