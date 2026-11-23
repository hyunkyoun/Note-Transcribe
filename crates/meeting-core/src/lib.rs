//! Hardware-independent logic for Note Transcribe.
//!
//! Everything in this crate is pure Rust with no audio, GPU, network or UI
//! dependencies so it can be unit-tested on any machine (`cargo test -p meeting-core`).

pub mod cluster;
pub mod discord_split;
pub mod echo;
pub mod hallucination;
pub mod labels;
pub mod merge;
pub mod prompt;
pub mod resample;
pub mod segment;
pub mod settings;
pub mod sse;
pub mod timeline;
pub mod vad;

pub use segment::{Segment, Source};
