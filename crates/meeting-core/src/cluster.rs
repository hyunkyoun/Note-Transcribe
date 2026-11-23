//! Speaker clustering on speaker-embedding vectors.
//!
//! * [`OnlineClusterer`] assigns each new utterance to a speaker immediately
//!   (used for live captions).
//! * [`refine_clusters`] re-clusters all utterances at the end of the meeting
//!   (k-means style reassignment + merging of near-duplicate speakers), which
//!   fixes most early mistakes made before the online centroids stabilized.

/// Cosine similarity; returns 0 for zero-length / mismatched vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

pub fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x = (*x as f64 / n) as f32;
        }
    }
}

/// Greedy online clustering with running-mean centroids.
#[derive(Debug, Clone)]
pub struct OnlineClusterer {
    threshold: f32,
    max_clusters: usize,
    centroids: Vec<Vec<f32>>,
    counts: Vec<usize>,
}

impl OnlineClusterer {
    pub fn new(threshold: f32, max_clusters: usize) -> Self {
        Self { threshold, max_clusters: max_clusters.max(1), centroids: Vec::new(), counts: Vec::new() }
    }

    pub fn num_clusters(&self) -> usize {
        self.centroids.len()
    }

    /// Assign an embedding to a cluster index (0-based), creating a new
    /// cluster if it is not similar enough to any existing one.
    pub fn assign(&mut self, embedding: &[f32]) -> usize {
        let mut e = embedding.to_vec();
        l2_normalize(&mut e);
        let best = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, cosine(c, &e)))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        match best {
            Some((i, sim)) if sim >= self.threshold || self.centroids.len() >= self.max_clusters => {
                self.update(i, &e);
                i
            }
            _ => {
                self.centroids.push(e);
                self.counts.push(1);
                self.centroids.len() - 1
            }
        }
    }

    fn update(&mut self, i: usize, e: &[f32]) {
        let n = self.counts[i] as f32;
        for (c, x) in self.centroids[i].iter_mut().zip(e) {
            *c = (*c * n + x) / (n + 1.0);
        }
        l2_normalize(&mut self.centroids[i]);
        self.counts[i] += 1;
    }
}

fn centroids(embs: &[Vec<f32>], labels: &[usize], k: usize) -> Vec<Option<Vec<f32>>> {
    let dim = embs.first().map(|e| e.len()).unwrap_or(0);
    let mut sums = vec![vec![0f32; dim]; k];
    let mut counts = vec![0usize; k];
    for (e, &l) in embs.iter().zip(labels) {
        let mut v = e.clone();
        l2_normalize(&mut v);
        for (s, x) in sums[l].iter_mut().zip(&v) {
            *s += x;
        }
        counts[l] += 1;
    }
    sums.into_iter()
        .zip(counts)
        .map(|(mut s, c)| {
            if c == 0 {
                None
            } else {
                l2_normalize(&mut s);
                Some(s)
            }
        })
        .collect()
}

/// Renumber labels 0.. in order of first appearance.
pub fn renumber_by_first_appearance(labels: &[usize]) -> Vec<usize> {
    let mut map: Vec<(usize, usize)> = Vec::new();
    labels
        .iter()
        .map(|l| {
            if let Some((_, n)) = map.iter().find(|(o, _)| o == l) {
                *n
            } else {
                let n = map.len();
                map.push((*l, n));
                n
            }
        })
        .collect()
}

/// End-of-meeting refinement. Starting from `initial` labels (e.g. from the
/// online clusterer), alternately (1) merge clusters whose centroids are at
/// least `threshold` similar and (2) reassign every embedding to its nearest
/// centroid, until stable. Returns labels renumbered by first appearance.
pub fn refine_clusters(embs: &[Vec<f32>], initial: &[usize], threshold: f32) -> Vec<usize> {
    assert_eq!(embs.len(), initial.len());
    if embs.is_empty() {
        return Vec::new();
    }
    let mut labels = renumber_by_first_appearance(initial);
    for _ in 0..20 {
        let k = labels.iter().max().map(|m| m + 1).unwrap_or(0);
        let cents = centroids(embs, &labels, k);

        // (1) Merge the most similar pair above threshold (one per iteration keeps it stable).
        let mut best: Option<(usize, usize, f32)> = None;
        for i in 0..k {
            for j in (i + 1)..k {
                if let (Some(a), Some(b)) = (&cents[i], &cents[j]) {
                    let s = cosine(a, b);
                    if s >= threshold && best.is_none_or(|(_, _, bs)| s > bs) {
                        best = Some((i, j, s));
                    }
                }
            }
        }
        let mut changed = false;
        if let Some((i, j, _)) = best {
            for l in labels.iter_mut() {
                if *l == j {
                    *l = i;
                }
            }
            changed = true;
        }

        // (2) Reassign to nearest centroid.
        let cents = centroids(embs, &labels, k);
        for (idx, e) in embs.iter().enumerate() {
            let nearest = cents
                .iter()
                .enumerate()
                .filter_map(|(c, v)| v.as_ref().map(|v| (c, cosine(v, e))))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(c, _)| c)
                .unwrap_or(labels[idx]);
            if nearest != labels[idx] {
                labels[idx] = nearest;
                changed = true;
            }
        }
        labels = renumber_by_first_appearance(&labels);
        if !changed {
            break;
        }
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random generator (no external deps).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        }
    }

    fn speaker_vec(seed: u64, dim: usize) -> Vec<f32> {
        let mut r = Lcg(seed);
        (0..dim).map(|_| r.next()).collect()
    }

    fn noisy(base: &[f32], noise: f32, r: &mut Lcg) -> Vec<f32> {
        base.iter().map(|x| x + noise * r.next()).collect()
    }

    #[test]
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
        assert_eq!(cosine(&[1.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn online_separates_distinct_speakers() {
        let dim = 64;
        let a = speaker_vec(1, dim);
        let b = speaker_vec(2, dim);
        let c = speaker_vec(3, dim);
        let mut r = Lcg(99);
        let mut cl = OnlineClusterer::new(0.5, 8);
        let mut got = Vec::new();
        for spk in [&a, &b, &a, &c, &b, &a, &c] {
            got.push(cl.assign(&noisy(spk, 0.15, &mut r)));
        }
        assert_eq!(got, vec![0, 1, 0, 2, 1, 0, 2]);
        assert_eq!(cl.num_clusters(), 3);
    }

    #[test]
    fn online_respects_max_clusters() {
        let mut cl = OnlineClusterer::new(0.99, 2);
        for s in 0..5 {
            cl.assign(&speaker_vec(s + 10, 32));
        }
        assert_eq!(cl.num_clusters(), 2);
    }

    #[test]
    fn refine_merges_split_speaker_and_fixes_mislabel() {
        let dim = 64;
        let a = speaker_vec(1, dim);
        let b = speaker_vec(2, dim);
        let mut r = Lcg(7);
        let embs: Vec<Vec<f32>> = [&a, &a, &b, &a, &b, &b, &a]
            .iter()
            .map(|s| noisy(s, 0.15, &mut r))
            .collect();
        // Online pass split speaker A into two clusters (0 and 2) and mislabelled index 5.
        let initial = vec![0, 0, 1, 2, 1, 0, 2];
        let refined = refine_clusters(&embs, &initial, 0.5);
        assert_eq!(refined, vec![0, 0, 1, 0, 1, 1, 0]);
    }

    #[test]
    fn refine_empty_and_single() {
        assert!(refine_clusters(&[], &[], 0.5).is_empty());
        assert_eq!(refine_clusters(&[vec![1.0, 2.0]], &[3], 0.5), vec![0]);
    }

    #[test]
    fn renumbering() {
        assert_eq!(renumber_by_first_appearance(&[5, 2, 5, 9, 2]), vec![0, 1, 0, 2, 1]);
    }
}
