# Models folder

Put the model files here (or point Settings → *Models folder* somewhere else).
The easiest way is `scripts\download-models.ps1`.

| File | Used for | Size |
|---|---|---|
| `ggml-large-v3-turbo.bin` | Whisper speech-to-text (default) | 1.6 GB |
| `ggml-base.bin` | Whisper fallback (CPU / missing main model) | 148 MB |
| `silero_vad.onnx` | Voice activity detection (skips silence) | 2 MB |
| `wespeaker_en_voxceleb_resnet34_LM.onnx` | Speaker embeddings for diarization | 26 MB |

Any other `ggml-<name>.bin` from <https://huggingface.co/ggerganov/whisper.cpp>
works too. Set Settings → *Model* to `<name>` (e.g. `large-v3-turbo-q5_0`, `small.en`).
