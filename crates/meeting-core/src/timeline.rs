//! Keeping the mic and loopback streams aligned to one wall clock.
//!
//! WASAPI loopback delivers *no packets at all* while nothing is playing, and
//! a device that is unplugged and re-opened restarts from zero. Both would
//! make one stream's sample count drift away from real time. [`Timeline`]
//! compares the number of samples received with the time elapsed since the
//! recording started and tells the caller to pad with silence (gap) or drop
//! samples (device clock running fast) so that sample index == wall time.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Adjust {
    /// Stream is on time.
    None,
    /// Insert this many zero samples *before* the batch.
    Pad(usize),
    /// Drop this many samples from the *start* of the batch.
    Drop(usize),
}

#[derive(Debug, Clone)]
pub struct Timeline {
    sample_rate: u32,
    received: u64,
    pad_threshold: u64,
    drop_threshold: u64,
}

impl Timeline {
    /// `pad_threshold_ms`: how far behind real time we tolerate before padding.
    /// `drop_threshold_ms`: how far ahead before dropping.
    pub fn new(sample_rate: u32, pad_threshold_ms: u32, drop_threshold_ms: u32) -> Self {
        Self {
            sample_rate,
            received: 0,
            pad_threshold: ms_to_samples(sample_rate, pad_threshold_ms as u64),
            drop_threshold: ms_to_samples(sample_rate, drop_threshold_ms as u64),
        }
    }

    /// Samples emitted so far (== current stream position).
    pub fn position(&self) -> u64 {
        self.received
    }

    /// Call when *no packets* have arrived for a while (WASAPI loopback goes
    /// completely quiet when nothing is playing). Returns how many zero
    /// samples to append so the stream keeps up with real time, leaving
    /// `margin_ms` of slack for a packet that may still be in flight.
    pub fn idle_fill(&mut self, elapsed_secs: f64, margin_ms: u32) -> usize {
        let expected = (elapsed_secs.max(0.0) * self.sample_rate as f64) as u64;
        if expected <= self.received + self.pad_threshold {
            return 0;
        }
        let target = expected.saturating_sub(ms_to_samples(self.sample_rate, margin_ms as u64));
        let n = target.saturating_sub(self.received);
        self.received += n;
        n as usize
    }

    pub fn position_ms(&self) -> u64 {
        self.received * 1000 / self.sample_rate as u64
    }

    /// Call for each batch. `elapsed_secs` is the wall time since recording
    /// start at the moment the *last* sample of the batch was captured.
    pub fn on_batch(&mut self, batch_len: usize, elapsed_secs: f64) -> Adjust {
        let expected_end = (elapsed_secs.max(0.0) * self.sample_rate as f64) as u64;
        let actual_end = self.received + batch_len as u64;
        if expected_end > actual_end + self.pad_threshold {
            let pad = expected_end - actual_end;
            self.received = expected_end;
            Adjust::Pad(pad as usize)
        } else if actual_end > expected_end + self.drop_threshold {
            let drop = ((actual_end - expected_end) as usize).min(batch_len);
            self.received += (batch_len - drop) as u64;
            Adjust::Drop(drop)
        } else {
            self.received = actual_end;
            Adjust::None
        }
    }
}

pub fn ms_to_samples(sample_rate: u32, ms: u64) -> u64 {
    ms * sample_rate as u64 / 1000
}

pub fn samples_to_ms(sample_rate: u32, samples: u64) -> u64 {
    samples * 1000 / sample_rate as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 16_000;

    #[test]
    fn on_time_stream_is_untouched() {
        let mut t = Timeline::new(SR, 250, 750);
        for i in 1..=100 {
            assert_eq!(t.on_batch(160, i as f64 * 0.01), Adjust::None);
        }
        assert_eq!(t.position(), 16_000);
    }

    #[test]
    fn late_start_is_padded() {
        let mut t = Timeline::new(SR, 250, 750);
        // First packet arrives 0.5 s after start, carrying 10 ms of audio.
        assert_eq!(t.on_batch(160, 0.5), Adjust::Pad(8_000 - 160));
        assert_eq!(t.position(), 8_000);
    }

    #[test]
    fn loopback_silence_gap_is_filled() {
        let mut t = Timeline::new(SR, 250, 750);
        for i in 1..=100 {
            t.on_batch(160, i as f64 * 0.01);
        }
        // Nothing played for 3 s, then a 10 ms packet.
        match t.on_batch(160, 4.01) {
            Adjust::Pad(n) => assert_eq!(n, 3 * 16_000),
            other => panic!("{other:?}"),
        }
        assert_eq!(t.position_ms(), 4_010);
    }

    #[test]
    fn small_jitter_is_tolerated() {
        let mut t = Timeline::new(SR, 250, 750);
        assert_eq!(t.on_batch(1_600, 0.1), Adjust::None);
        assert_eq!(t.on_batch(1_600, 0.35), Adjust::None); // 150 ms late: OK
    }

    #[test]
    fn fast_clock_drops_excess() {
        let mut t = Timeline::new(SR, 250, 750);
        // 2 s of audio delivered after only 1 s of wall time.
        assert_eq!(t.on_batch(32_000, 1.0), Adjust::Drop(16_000));
        assert_eq!(t.position(), 16_000);
    }

    #[test]
    fn idle_fill_keeps_up_with_wall_clock() {
        let mut t = Timeline::new(SR, 250, 750);
        assert_eq!(t.on_batch(1_600, 0.1), Adjust::None);
        // Only 200 ms behind: within tolerance, nothing to fill yet.
        assert_eq!(t.idle_fill(0.3, 100), 0);
        // 1 s of total silence (no packets): fill up to now minus the margin.
        assert_eq!(t.idle_fill(1.1, 100), 16_000 - 1_600);
        assert_eq!(t.position_ms(), 1_000);
        // A packet that arrives right after is on time (no double padding).
        assert_eq!(t.on_batch(1_600, 1.1), Adjust::None);
        assert_eq!(t.position_ms(), 1_100);
    }

    #[test]
    fn conversions() {
        assert_eq!(ms_to_samples(SR, 1_500), 24_000);
        assert_eq!(samples_to_ms(SR, 24_000), 1_500);
    }
}
