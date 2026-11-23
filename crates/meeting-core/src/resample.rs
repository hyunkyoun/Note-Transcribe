//! Down-mixing and streaming sample-rate conversion to 16 kHz mono for Whisper.
//!
//! A 4th-order Butterworth low-pass (two cascaded biquads) removes content
//! above ~0.45 * output rate before linear interpolation, which is plenty for
//! speech recognition and keeps us free of heavy DSP dependencies.

/// Average interleaved channels into mono.
pub fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|f| f.iter().sum::<f32>() / channels as f32)
        .collect()
}

#[derive(Debug, Clone, Copy, Default)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    fn lowpass(sample_rate: f64, cutoff: f64, q: f64) -> Self {
        let w0 = 2.0 * std::f64::consts::PI * cutoff / sample_rate;
        let alpha = w0.sin() / (2.0 * q);
        let cos = w0.cos();
        let a0 = 1.0 + alpha;
        Self {
            b0: (1.0 - cos) / 2.0 / a0,
            b1: (1.0 - cos) / a0,
            b2: (1.0 - cos) / 2.0 / a0,
            a1: -2.0 * cos / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f64) -> f64 {
        // Transposed direct form II.
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// Streaming mono resampler. Feed arbitrary-sized chunks; state carries over.
#[derive(Debug, Clone)]
pub struct StreamResampler {
    in_rate: u32,
    out_rate: u32,
    step: f64,
    /// Position of the next output sample, in input samples relative to the
    /// start of the next chunk (may be in [-1, 0): between `prev` and chunk[0]).
    t: f64,
    prev: f32,
    filters: Vec<Biquad>,
}

impl StreamResampler {
    pub fn new(in_rate: u32, out_rate: u32) -> Self {
        assert!(in_rate > 0 && out_rate > 0);
        let filters = if out_rate < in_rate {
            let cutoff = 0.45 * out_rate as f64;
            vec![
                Biquad::lowpass(in_rate as f64, cutoff, 0.541_196_1),
                Biquad::lowpass(in_rate as f64, cutoff, 1.306_563),
            ]
        } else {
            Vec::new()
        };
        Self { in_rate, out_rate, step: in_rate as f64 / out_rate as f64, t: 0.0, prev: 0.0, filters }
    }

    pub fn in_rate(&self) -> u32 {
        self.in_rate
    }

    pub fn out_rate(&self) -> u32 {
        self.out_rate
    }

    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }
        if self.in_rate == self.out_rate {
            return input.to_vec();
        }
        let x: Vec<f32> = if self.filters.is_empty() {
            input.to_vec()
        } else {
            input
                .iter()
                .map(|&s| {
                    let mut v = s as f64;
                    for f in self.filters.iter_mut() {
                        v = f.process(v);
                    }
                    v as f32
                })
                .collect()
        };
        let len = x.len() as isize;
        let mut out = Vec::with_capacity((x.len() as f64 / self.step) as usize + 2);
        loop {
            let i = self.t.floor() as isize;
            if i + 1 >= len {
                break;
            }
            let f = (self.t - i as f64) as f32;
            let s0 = if i < 0 { self.prev } else { x[i as usize] };
            let s1 = x[(i + 1) as usize];
            out.push(s0 + (s1 - s0) * f);
            self.t += self.step;
        }
        self.t -= len as f64;
        self.prev = x[x.len() - 1];
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, rate: u32, secs: f32) -> Vec<f32> {
        (0..(rate as f32 * secs) as usize)
            .map(|n| (2.0 * std::f32::consts::PI * freq * n as f32 / rate as f32).sin())
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    /// Estimate frequency from zero crossings.
    fn zc_freq(x: &[f32], rate: u32) -> f32 {
        let zc = x.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count();
        zc as f32 / 2.0 / (x.len() as f32 / rate as f32)
    }

    #[test]
    fn downmix_stereo() {
        assert_eq!(downmix(&[1.0, 0.0, 0.5, 0.5], 2), vec![0.5, 0.5]);
        assert_eq!(downmix(&[0.3, 0.4], 1), vec![0.3, 0.4]);
    }

    #[test]
    fn output_length_matches_ratio_across_chunks() {
        let mut r = StreamResampler::new(48_000, 16_000);
        let input = sine(440.0, 48_000, 2.0);
        let mut out = Vec::new();
        for chunk in input.chunks(441) {
            out.extend(r.process(chunk));
        }
        let expected = input.len() / 3;
        assert!((out.len() as i64 - expected as i64).abs() <= 2, "{} vs {expected}", out.len());
    }

    #[test]
    fn preserves_speech_band_tone() {
        for in_rate in [44_100u32, 48_000, 96_000] {
            let mut r = StreamResampler::new(in_rate, 16_000);
            let out = r.process(&sine(1_000.0, in_rate, 1.0));
            let tail = &out[1600..]; // skip filter warm-up
            let f = zc_freq(tail, 16_000);
            assert!((f - 1_000.0).abs() < 15.0, "freq {f} at {in_rate}");
            assert!((rms(tail) - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.05);
        }
    }

    #[test]
    fn attenuates_content_above_nyquist() {
        let mut r = StreamResampler::new(48_000, 16_000);
        let out = r.process(&sine(12_000.0, 48_000, 1.0));
        assert!(rms(&out[1600..]) < 0.08, "aliasing too strong: {}", rms(&out[1600..]));
    }

    #[test]
    fn passthrough_and_upsample() {
        let mut same = StreamResampler::new(16_000, 16_000);
        assert_eq!(same.process(&[0.1, 0.2]), vec![0.1, 0.2]);
        let mut up = StreamResampler::new(8_000, 16_000);
        let out = up.process(&sine(500.0, 8_000, 1.0));
        assert!((out.len() as i64 - 16_000).abs() <= 2);
    }
}
