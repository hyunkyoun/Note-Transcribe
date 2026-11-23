//! Energy-based voice activity detector.
//!
//! Used only as a fallback when the Silero VAD model (`silero_vad.onnx`) is
//! not installed. Silero is far more robust to noise and music; this keeps the
//! app usable with nothing but a Whisper model.

use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct EnergyVad {
    frame: usize,
    hangover: u64,
    min_speech: usize,
    max_speech: usize,
    pre_roll: usize,
    min_abs: f32,
    mult: f32,
    floor: Option<f32>,
    pos: u64,
    pending: Vec<f32>,
    history: VecDeque<f32>,
    in_speech: bool,
    seg_start: u64,
    voice_start: u64,
    last_voice: u64,
    tail: u64,
    buf: Vec<f32>,
    ready: VecDeque<(u64, Vec<f32>)>,
}

impl EnergyVad {
    pub fn new(sample_rate: u32) -> Self {
        let ms = |m: u32| (sample_rate as u64 * m as u64 / 1000) as usize;
        Self {
            frame: ms(30),
            hangover: ms(500) as u64,
            min_speech: ms(250),
            max_speech: ms(12_000),
            pre_roll: ms(200),
            min_abs: 0.006,
            mult: 3.0,
            floor: None,
            pos: 0,
            pending: Vec::new(),
            history: VecDeque::new(),
            in_speech: false,
            seg_start: 0,
            voice_start: 0,
            last_voice: 0,
            tail: ms(200) as u64,
            buf: Vec::new(),
            ready: VecDeque::new(),
        }
    }

    pub fn is_speech(&self) -> bool {
        self.in_speech
    }

    pub fn accept(&mut self, samples: &[f32]) {
        self.pending.extend_from_slice(samples);
        while self.pending.len() >= self.frame {
            let frame: Vec<f32> = self.pending.drain(..self.frame).collect();
            self.process_frame(&frame);
        }
    }

    fn process_frame(&mut self, frame: &[f32]) {
        let rms = (frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32).sqrt();
        let floor = *self.floor.get_or_insert(rms.max(1e-4));
        let voiced = rms > (floor * self.mult).max(self.min_abs);
        let frame_start = self.pos;
        let frame_end = self.pos + frame.len() as u64;

        if !self.in_speech {
            // Track background level only while not speaking.
            self.floor = Some((floor * 0.95 + rms * 0.05).max(1e-4));
            if voiced {
                self.in_speech = true;
                let pre: Vec<f32> = self.history.iter().copied().collect();
                self.seg_start = frame_start - pre.len() as u64;
                self.buf = pre;
                self.buf.extend_from_slice(frame);
                self.voice_start = frame_start;
                self.last_voice = frame_end;
            }
        } else {
            self.buf.extend_from_slice(frame);
            if voiced {
                self.last_voice = frame_end;
            }
            if frame_end - self.last_voice > self.hangover {
                self.finish();
            } else if self.buf.len() >= self.max_speech {
                let start = self.seg_start;
                let seg = std::mem::take(&mut self.buf);
                self.seg_start = start + seg.len() as u64;
                self.voice_start = self.seg_start;
                self.ready.push_back((start, seg));
            }
        }

        self.history.extend(frame.iter().copied());
        while self.history.len() > self.pre_roll {
            self.history.pop_front();
        }
        self.pos = frame_end;
    }

    fn finish(&mut self) {
        self.in_speech = false;
        let mut seg = std::mem::take(&mut self.buf);
        // Trim trailing silence (keep a short tail) and require enough *voiced* audio.
        let keep = (self.last_voice + self.tail).saturating_sub(self.seg_start) as usize;
        seg.truncate(keep.max(1));
        let voiced = self.last_voice.saturating_sub(self.voice_start) as usize;
        if voiced >= self.min_speech {
            self.ready.push_back((self.seg_start, seg));
        }
    }

    /// Next completed `(start_sample, samples)` speech segment.
    pub fn pop(&mut self) -> Option<(u64, Vec<f32>)> {
        self.ready.pop_front()
    }

    /// End of stream: close any open segment.
    pub fn flush(&mut self) {
        if self.in_speech {
            self.finish();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 16_000;

    fn noise(n: usize, amp: f32, seed: &mut u64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((*seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * amp
            })
            .collect()
    }

    fn tone(n: usize, amp: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.2).sin() * amp).collect()
    }

    #[test]
    fn detects_one_burst() {
        let mut seed = 1;
        let mut v = EnergyVad::new(SR);
        v.accept(&noise(SR as usize, 0.002, &mut seed)); // 1 s quiet
        assert!(!v.is_speech());
        v.accept(&tone(2 * SR as usize, 0.3)); // 2 s speech
        assert!(v.is_speech());
        v.accept(&noise(SR as usize, 0.002, &mut seed)); // 1 s quiet
        let (start, seg) = v.pop().expect("segment");
        assert!(v.pop().is_none());
        let start_s = start as f32 / SR as f32;
        let len_s = seg.len() as f32 / SR as f32;
        assert!((0.75..=1.0).contains(&start_s), "start {start_s}");
        assert!((2.0..=2.8).contains(&len_s), "len {len_s}");
    }

    #[test]
    fn ignores_short_clicks() {
        let mut seed = 2;
        let mut v = EnergyVad::new(SR);
        v.accept(&noise(SR as usize, 0.002, &mut seed));
        v.accept(&tone(480, 0.5)); // 30 ms click (+pre-roll < min_speech)
        v.accept(&noise(SR as usize, 0.002, &mut seed));
        assert!(v.pop().is_none());
    }

    #[test]
    fn splits_long_speech_and_flushes() {
        let mut v = EnergyVad::new(SR);
        v.accept(&vec![0.0; 1600]);
        v.accept(&tone(30 * SR as usize, 0.3));
        v.flush();
        let mut segs = Vec::new();
        while let Some(s) = v.pop() {
            segs.push(s);
        }
        assert!(segs.len() >= 3);
        assert!(segs.iter().all(|(_, s)| s.len() <= 12 * SR as usize + 480));
        // Contiguous: each segment starts where the previous ended.
        for w in segs.windows(2) {
            assert_eq!(w[0].0 + w[0].1.len() as u64, w[1].0);
        }
    }
}
