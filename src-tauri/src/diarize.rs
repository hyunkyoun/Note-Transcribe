//! Speaker diarization: sherpa-onnx speaker embeddings (CPU) + online
//! clustering for live labels, then a refinement pass at the end.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use meeting_core::cluster::{refine_clusters, OnlineClusterer};
use parking_lot::Mutex;
use sherpa_rs::speaker_id::{EmbeddingExtractor, ExtractorConfig};

use crate::vad::SAMPLE_RATE;

pub type SharedExtractor = Arc<Mutex<EmbeddingExtractor>>;

/// Below this much audio an embedding is unreliable; reuse the previous speaker.
const MIN_EMBED_SAMPLES: usize = SAMPLE_RATE as usize; // 1.0 s
/// Cap the audio we embed (middle part of long segments) to bound CPU time.
const MAX_EMBED_SAMPLES: usize = 10 * SAMPLE_RATE as usize;

pub fn load_extractor(models_dir: &Path) -> Result<SharedExtractor, String> {
    let Some(path) = crate::paths::speaker_model_path(models_dir) else {
        return Err(format!(
            "no speaker-embedding model (*.onnx) in {}; all desktop audio will be \"Speaker 1\"",
            models_dir.display()
        ));
    };
    let threads = std::thread::available_parallelism().map(|n| n.get().min(4)).unwrap_or(2);
    let config = ExtractorConfig {
        model: path.to_string_lossy().to_string(),
        provider: None, // CPU: small model, keeps the GPU free for Whisper
        num_threads: Some(threads),
        debug: false,
    };
    EmbeddingExtractor::new(config)
        .map(|e| Arc::new(Mutex::new(e)))
        .map_err(|e| format!("speaker model failed to load: {e}"))
}

struct Record {
    id: u64,
    embedding: Option<Vec<f32>>,
    online: usize,
}

pub struct Diarizer {
    extractor: Option<SharedExtractor>,
    online: OnlineClusterer,
    records: Vec<Record>,
    last: Option<usize>,
    threshold: f32,
}

impl Diarizer {
    pub fn new(extractor: Option<SharedExtractor>, threshold: f32, max_speakers: usize) -> Self {
        Self { extractor, online: OnlineClusterer::new(threshold, max_speakers), records: Vec::new(), last: None, threshold }
    }

    /// Live cluster for a finished segment.
    pub fn assign(&mut self, id: u64, samples: &[f32]) -> usize {
        let embedding = match &self.extractor {
            Some(ex) if samples.len() >= MIN_EMBED_SAMPLES => {
                let slice = if samples.len() > MAX_EMBED_SAMPLES {
                    let off = (samples.len() - MAX_EMBED_SAMPLES) / 2;
                    &samples[off..off + MAX_EMBED_SAMPLES]
                } else {
                    samples
                };
                match ex.lock().compute_speaker_embedding(slice.to_vec(), SAMPLE_RATE) {
                    Ok(e) if !e.is_empty() => Some(e),
                    Ok(_) => None,
                    Err(e) => {
                        log::warn!("speaker embedding failed: {e}");
                        None
                    }
                }
            }
            _ => None,
        };
        let cluster = match &embedding {
            Some(e) => self.online.assign(e),
            None => self.last.unwrap_or(0),
        };
        self.last = Some(cluster);
        self.records.push(Record { id, embedding, online: cluster });
        cluster
    }

    /// Final segment-id -> cluster mapping after end-of-meeting refinement.
    pub fn refined(&self) -> HashMap<u64, usize> {
        let with: Vec<&Record> = self.records.iter().filter(|r| r.embedding.is_some()).collect();
        let embs: Vec<Vec<f32>> = with.iter().map(|r| r.embedding.clone().unwrap_or_default()).collect();
        let initial: Vec<usize> = with.iter().map(|r| r.online).collect();
        let refined = refine_clusters(&embs, &initial, self.threshold);
        let by_id: HashMap<u64, usize> = with.iter().zip(refined).map(|(r, l)| (r.id, l)).collect();

        // Segments without an embedding inherit the label of the previous one that had one.
        let mut out = HashMap::new();
        let mut prev: Option<usize> = None;
        for r in &self.records {
            let l = by_id.get(&r.id).copied().or(prev).unwrap_or(0);
            if by_id.contains_key(&r.id) {
                prev = Some(l);
            }
            out.insert(r.id, l);
        }
        out
    }
}
