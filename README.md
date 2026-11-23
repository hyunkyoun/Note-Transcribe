# Note Transcribe

A lightweight Windows 11 desktop app that **passively** transcribes your meetings. It does not join the call as a participant or bot. It only listens to audio on your own machine:

- **Desktop audio** via WASAPI loopback (whatever your speakers or headphones play, i.e. the other participants). No virtual audio cable is needed.
- **Your microphone**, as a normal capture stream.

It shows live, speaker-labelled captions while you talk. When you stop, you can rename speakers, and then it sends the transcript to Claude for a summary (using a prompt file you can edit). Finally it posts the summary and transcript to a Discord channel through a webhook.

> **Privacy and consent.** You are responsible for telling meeting participants that you are recording and transcribing, and for following the recording and consent laws where you and they are located, as well as your company's policies. Audio is processed locally. Only the text transcript is sent to the Anthropic API (for the summary) and to the Discord channel you configure.

---

## Contents

1. [How it works](#how-it-works)
2. [Stack choices and trade-offs](#stack-choices-and-trade-offs)
3. [Prerequisites (Windows 11)](#prerequisites-windows-11)
4. [Setup (PowerShell)](#setup-powershell)
5. [Build and run](#build-and-run)
6. [Using the app](#using-the-app)
7. [Editing the summary prompt](#editing-the-summary-prompt)
8. [Creating a Discord webhook](#creating-a-discord-webhook)
9. [Where files live](#where-files-live)
10. [Resource use](#resource-use)
11. [Troubleshooting](#troubleshooting)
12. [Tests](#tests)
13. [Project structure](#project-structure)

---

## How it works

```
 Output device ──WASAPI loopback──┐                       ┌─ Silero VAD ─ speaker embedding ─ online clustering ─┐
                                  ├─ mono ─ 16 kHz ─ align┤                                                       ├─► Whisper (GPU) ─► captions
 Microphone ───WASAPI capture─────┘   (per stream)        └─ Silero VAD ─ (diarize only if "Mic is only me" is off)┘        │
                                                                                                                             ▼
                              Stop ─► re-cluster speakers ─► drop mic echo ─► rename table ─► Claude summary ─► Discord
```

- **Two independent streams.** Each source has its own thread that downmixes, resamples to 16 kHz mono, and aligns it to one shared wall clock. WASAPI loopback sends *nothing* while the output is silent, so the gaps are filled with silence to keep both timelines in sync. Each stream then runs its own VAD and diarization.
- **One shared Whisper model.** A single GPU worker transcribes finished utterances ("finals") first. When it has spare time it re-transcribes the utterance still in progress every ~1 s to give a **live partial caption** (shown in grey italics). That keeps captions roughly **1–2 s behind speech** on an RTX-class GPU.
- **Merging.** Final segments from both streams are timestamped against the same clock. They are written to disk as they arrive and merged into one time-ordered transcript.
- **Speaker labels.**
  - Desktop audio is split into `Speaker 1`, `Speaker 2`, …
  - Microphone audio:
    - **Mic is only me = ON (default):** every mic segment gets your name (or `Me`), and no mic diarization runs.
    - **Mic is only me = OFF:** the mic is diarized separately into `Mic Speaker 1`, `Mic Speaker 2`, … These labels never mix with the desktop speakers.
- **Rename before summarizing.** After Stop, a table lets you map `Speaker 2 → Dana` and so on. Renames apply to the saved transcript, the Claude prompt and the Discord post.

### Transcription: whisper.cpp (via `whisper-rs`) with CUDA

| | whisper.cpp (chosen) | faster-whisper |
|---|---|---|
| Packaging on Windows | Compiled into the exe. No Python. | Needs Python, CTranslate2 and cuDNN DLLs, run as a sidecar process |
| GPU latency (turbo model, ~5 s clip) | ~100–300 ms | Similar |
| Accuracy | Same Whisper weights | Same Whisper weights |
| Idle memory | Nothing loaded until Start | An always-on Python process |

The model is configurable in Settings and defaults to `large-v3-turbo`. A **fallback model** (default `base`) is used when the main file is missing. It is also used automatically, on CPU, when CUDA isn't available. Hallucinations are kept down by:

- the VAD, so Whisper never sees pure silence;
- Whisper's own no-speech probability;
- a filter for stock phrases ("Thanks for watching!") and repetition loops;
- `no_context` decoding.

### Diarization: sherpa-onnx speaker embeddings (no Python, no token)

Each utterance (≥ 1 s) gets a speaker embedding from a small ONNX model (WeSpeaker ResNet34, 26 MB), computed on the **CPU** in ~20–60 ms. The GPU stays free for Whisper. Embeddings are clustered **online** (cosine similarity against running centroids) so labels appear live. At Stop, a **refinement pass** re-assigns every utterance using the whole meeting's evidence and merges duplicate speakers. Labels in the saved transcript can therefore be slightly better than the live captions.

Why not pyannote? Its pipeline is more accurate on overlapping speech, but it needs a Python process, PyTorch and a gated Hugging Face model (token required). sherpa-onnx gives most of the benefit with none of that. `HF_TOKEN` exists in `.env` only in case you later swap in a gated model.

Accuracy notes:
- It works best for distinct voices that each speak in turns of a second or more.
- Two people talking over each other within one utterance are labelled as one speaker.
- If people get merged, raise **Settings → Same-speaker threshold** (0.5 is the default). If one person gets split into several labels, lower it.
- On desktop audio, everyone in a meeting app may be compressed by the same codec, which makes voices more alike than in person. Expect the occasional mislabel, and fix it with the rename table.

---

## Stack choices and trade-offs

**Tauri v2 (Rust + WebView2) with a plain TypeScript frontend**, as you preferred. A .NET 8 WinUI 3 or WPF app would also be lightweight and has first-class WASAPI APIs. However, Whisper on CUDA from .NET means P/Invoke into whisper.cpp anyway (Whisper.net), and Tauri keeps the UI simpler to change. Tauri uses the WebView2 runtime that ships with Windows 11 instead of bundling Chromium. The app process idles at tens of MB, compared with 150 MB or more for Electron.

Other decisions worth knowing:

- **Audio:** `cpal` on WASAPI. Opening an input stream on a render device sets `AUDCLNT_STREAMFLAGS_LOOPBACK`.
- **HTTPS:** `reqwest` with `native-tls`, i.e. Windows SChannel and the OS certificate store. That works with corporate proxies or inspection certificates, and needs no C toolchain for TLS.
- **Claude API:** there is no official Anthropic SDK for Rust, so the summary uses the Messages API over raw HTTPS. Responses are **streamed** (no timeouts on long outputs). Retries use exponential backoff and honor `retry-after`. On models that support it, `fallbacks: "default"` is sent so a safety-classifier refusal is retried on a fallback model server-side; set `CLAUDE_FALLBACKS=off` to disable this.
- **Long meetings:** if the transcript is estimated above `CLAUDE_MAX_INPUT_TOKENS` (default 150k), or the API says the prompt is too long, it is split on line boundaries and each part is turned into detailed notes. Those notes are then fed through **your** prompt template, so the final format is still controlled by your file.

**Deviations from the spec:**
- **Summary trigger.** The spec says to summarize on Stop *and* to allow renaming before the summary, and those conflict. So **Stop** finalizes the transcript and opens the rename table, and **Summarize & send to Discord** does the rest. If you prefer one click, turn on **Settings → Summarize automatically on Stop**.
- **Echo suppression (extra feature).** If you use laptop speakers instead of headphones, your mic hears the other participants. The app drops mic segments that duplicate overlapping desktop segments. You can turn this off in Settings.

---

## Prerequisites (Windows 11)

Install these once, in an **elevated** PowerShell where noted:

| Tool | Why | Install |
|---|---|---|
| **Visual Studio 2022 Build Tools** with *Desktop development with C++* (MSVC + Windows 11 SDK) | Rust's MSVC linker; compiles whisper.cpp | `winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive"` |
| **Rust** (stable, MSVC toolchain) | Backend | `winget install Rustlang.Rustup`, then `rustup default stable-x86_64-pc-windows-msvc` |
| **Node.js 20+** | Frontend tooling and Tauri CLI | `winget install OpenJS.NodeJS.LTS` |
| **CMake 3.24+** | Builds whisper.cpp | `winget install Kitware.CMake` |
| **LLVM / Clang** | `bindgen` generates the whisper.cpp and sherpa-onnx bindings | `winget install LLVM.LLVM`, then set `LIBCLANG_PATH` (see below) |
| **NVIDIA driver** (recent) | GPU | GeForce Experience / nvidia.com |
| **CUDA Toolkit 12.8 or newer** | **Required for the RTX 5070 Ti** (Blackwell, sm_120 needs CUDA ≥ 12.8) | `winget install Nvidia.CUDA` or https://developer.nvidia.com/cuda-downloads |
| **WebView2 Runtime** | UI | Preinstalled on Windows 11 |

Set `LIBCLANG_PATH` once (new terminals pick it up):

```powershell
setx LIBCLANG_PATH "C:\Program Files\LLVM\bin"
```

Check the toolchain (open a **new** terminal first):

```powershell
rustc --version; cargo --version; node --version; cmake --version; nvcc --version; $env:CUDA_PATH
```

`nvcc --version` must report **12.8 or newer**. `CUDA_PATH` must point at it (the CUDA installer sets this).

---

## Setup (PowerShell)

```powershell
git clone https://github.com/hyunkyoun/Note-Transcribe.git
cd Note-Transcribe

# 1) Secrets: copy the template, then paste your keys into .env
Copy-Item .env.example .env
notepad .env        # set ANTHROPIC_API_KEY, CLAUDE_MODEL, DISCORD_WEBHOOK_URL

# 2) Models (about 1.8 GB total): Whisper large-v3-turbo + base fallback, Silero VAD, speaker embeddings
powershell -ExecutionPolicy Bypass -File .\scripts\download-models.ps1

# 3) Frontend dependencies
npm install
```

`.env` is git-ignored. In it:

- `ANTHROPIC_API_KEY`: create one at https://console.anthropic.com/settings/keys.
- `CLAUDE_MODEL`: defaults to `claude-opus-5-5`. Any Messages-API model ID works.
- `DISCORD_WEBHOOK_URL`: see [Creating a Discord webhook](#creating-a-discord-webhook).
- `HF_TOKEN`: optional and not used by default.

Optional tuning variables are documented in `.env.example`. After editing `.env` while the app runs, click **Settings → Reload .env**; no restart is needed.

### Download the models (manual alternative)

Put these files in `models\` (or choose another folder in **Settings → Models folder**):

| File | Source |
|---|---|
| `ggml-large-v3-turbo.bin` | https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin |
| `ggml-base.bin` (CPU fallback) | https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin |
| `silero_vad.onnx` | https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx |
| `wespeaker_en_voxceleb_resnet34_LM.onnx` | https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx |

Other Whisper sizes work too: put `ggml-<name>.bin` in the folder and type `<name>` into **Settings → Model**. For example, `large-v3-turbo-q5_0` uses about half the VRAM, and `small.en` is good for CPU-only use. The app still runs without the VAD or speaker model, but quality is lower (basic energy VAD, no speaker separation).

---

## Build and run

```powershell
npm run tauri dev       # development: hot-reloads the UI, rebuilds Rust on change
npm run tauri build     # release installer: target\release\bundle\nsis\Note Transcribe_0.1.0_x64-setup.exe
```

The first build compiles whisper.cpp with CUDA, which takes **10–20 minutes**. Later builds are incremental.

- **Faster CUDA builds.** By default whisper.cpp compiles several GPU architectures. To build only for the 5070 Ti (Blackwell):

  ```powershell
  $env:CMAKE_CUDA_ARCHITECTURES = "120a-real"
  npm run tauri build
  ```

- **CPU-only build** (no CUDA toolkit needed; uses the fallback model):

  ```powershell
  npm run dev:cpu
  npm run build:cpu
  ```

- **Bundled DLLs.** `npm run tauri build` copies the sherpa-onnx runtime DLLs (`sherpa-onnx-c-api.dll`, `onnxruntime.dll`) into the installer automatically, via `scripts/copy-dlls.mjs`. The CUDA runtime DLLs come from your CUDA Toolkit, which is on `PATH`. If you install the app on a machine *without* the toolkit, copy `cudart64_12.dll`, `cublas64_12.dll` and `cublasLt64_12.dll` from `%CUDA_PATH%\bin` next to `note-transcribe.exe`.

---

## Using the app

1. **Pick sources** on the main view.
   - **Desktop audio:** the output device your meeting plays through. *System default* follows Windows when you switch devices.
   - **Microphone.**
   - Untick either one to capture only the other. The choices are saved.
2. Press **● Start**.
   - The Whisper model loads in 1–3 s; audio is buffered meanwhile.
   - Captions stream in. A grey italic line is the utterance in progress.
   - The red indicator shows elapsed time.
3. Press **■ Stop**. The rest of the queue is transcribed, speakers are re-clustered, and the transcript is saved.
4. In the **rename table**, type real names (e.g. `Speaker 2 → Dana`, `Mic Speaker 1 → Eun`).
5. Click **Summarize & send to Discord**.
   - The summary is shown in the app and saved as `<id>.summary.md`.
   - Then it is posted to Discord: summary first, then the transcript (attached as a `.txt` file if it's long).
6. Other buttons:
   - **Re-send last summary to Discord** works for the last meeting, even after a restart.
   - **Send transcript only** posts without a summary.

**If the summary fails** (no key, network down, API outage): the app retries with backoff. If it still fails, the error is shown and **the transcript is already safe on disk**. Fix the problem and click Summarize again.

---

## Editing the summary prompt

The prompt is a plain Markdown file, **`prompts/summary_prompt.md`**. It is re-read every time you summarize, so edits apply immediately with no rebuild or restart. Open it from **Settings → Edit summary prompt**.

Placeholders:

| Placeholder | Replaced with |
|---|---|
| `{{TRANSCRIPT}}` | The full transcript, one turn per line: `[HH:MM:SS] Name: text` |
| `{{DATE}}` | Meeting start, e.g. `2026-11-23 14:05` |
| `{{DURATION}}` | e.g. `42m 10s` or `1h 05m` |
| `{{SPEAKERS}}` | Comma-separated speaker names (after renames) |

- If you remove `{{TRANSCRIPT}}`, the transcript is appended at the end so it's never lost.
- The default prompt produces: TL;DR, attendees, key discussion points, decisions, action items with owners, and open questions.
- To keep several prompts, point `SUMMARY_PROMPT_PATH` in `.env` at another file.
- In the installed app, the file is in the install folder under `prompts\summary_prompt.md` (per-user install, so no admin rights are needed to edit it). Settings shows the exact path.

---

## Creating a Discord webhook

1. In Discord, open the channel and go to **Edit Channel → Integrations → Webhooks → New Webhook**.
2. Name it (e.g. "Meeting Notes") and click **Copy Webhook URL**.
3. Paste it into `.env` as `DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...`, then click **Reload .env**.

How posting behaves:

- Messages are split cleanly at 2,000 characters. Splits happen at paragraph, line, sentence or word boundaries, and code fences are kept balanced.
- Mentions are disabled, so an `@everyone` in a transcript can't ping the channel.
- Rate limits (HTTP 429) are retried automatically.

---

## Where files live

| What | Location |
|---|---|
| Settings | `%APPDATA%\NoteTranscribe\settings.json` |
| Transcripts (default) | `%APPDATA%\NoteTranscribe\transcripts\` (change in Settings) |
| Log | `%APPDATA%\NoteTranscribe\logs\app.log` (never contains keys or webhook URLs) |
| Secrets | `.env` in the repo root (dev) or `%APPDATA%\NoteTranscribe\.env` (installed app) |
| Models | `models\` in the repo root, next to the installed exe, or the folder set in Settings |

**`settings.json` behavior:**
- **When it's written:** on every change (debounced 0.5 s, written atomically) and again when the window closes.
- **What it holds:** your name, the "Mic is only me" toggle, devices, model, language, raw-audio option, font size, etc. It never holds secrets.
- **If it's missing or corrupt:** the app uses defaults. A corrupt file is renamed to `settings.corrupt-<time>.json` so you can inspect it, and any still-valid fields are kept.

**Files per meeting** (`<id>` = `YYYY-MM-DD_HHMM`):

| File | Contents |
|---|---|
| `<id>.jsonl` | **Crash-safe log**: each final segment is appended and flushed to disk as it's transcribed |
| `<id>.txt` | Readable transcript: written live, then rewritten with final speaker labels and renames |
| `<id>.json` | Final segments, renames and summary (used by Re-send) |
| `<id>.summary.md` | The Claude summary |
| `<id>.desktop.wav`, `<id>.mic.wav` | Only if **Save raw audio** is on (off by default); 16 kHz mono |

If you close the window mid-meeting, the app spends up to 5 s transcribing what's still queued, then finalizes and saves everything transcribed so far.

---

## Resource use

These are approximate estimates, not measured on your machine. Check Task Manager.

| State | RAM | VRAM |
|---|---|---|
| Idle (window open) | ~40 MB app + ~80–120 MB WebView2 processes | 0 |
| Recording, `large-v3-turbo` on CUDA | +200–350 MB (VAD, speaker model, buffers) | ~1.8–2.5 GB (model + compute buffers + CUDA context) |
| Recording, `large-v3-turbo-q5_0` on CUDA | same | ~1.0–1.4 GB |
| Recording, `base` on CPU | ~+400 MB | 0 |

- **After Stop**, the Whisper model and speaker model are **freed** by default. The CUDA driver may keep a small context (~300 MB VRAM) until the app exits. Enable **Keep Whisper model in memory** for instant starts at the cost of holding VRAM.
- **GPU load** is low: each utterance is a short burst (~100–300 ms on a 5070 Ti) plus a partial refresh roughly every second while someone talks.
- **CPU load** is a few percent (resampling, VAD, embeddings).

---

## Troubleshooting

**No desktop audio / only my mic shows up**
- Pick the output device your meeting app actually plays through. With headphones, choose the headphones, or leave *System default* if Windows switches to them.
- Some apps use **exclusive mode** or protected (DRM) audio, which loopback can't capture. In the meeting app's audio settings, use the normal Windows output device. In *Sound → device Properties → Advanced*, untick "Allow applications to take exclusive control".
- Bluetooth headsets in **hands-free** mode (when their mic is in use) expose a different output device ("Headset" rather than "Headphones"). Select that one.
- "Device stopped (unplugged or changed); reconnecting…" is normal when you change devices. Capture resumes automatically.

**Microphone is silent**
- Check *Settings → Privacy & security → Microphone → Let desktop apps access your microphone*.

**"CUDA GPU not available; using CPU with the fallback model"**
- Confirm `nvcc --version` is **≥ 12.8** and `nvidia-smi` shows the GPU.
- If you installed CUDA after the first build, rebuild whisper.cpp:

  ```powershell
  cargo clean -p whisper-rs-sys
  npm run tauri dev
  ```

- To build only for Blackwell, set `$env:CMAKE_CUDA_ARCHITECTURES="120a-real"` before building.

**App won't start: `cudart64_12.dll` / `cublas64_12.dll` not found**
- The CUDA build links the CUDA runtime. Make sure `%CUDA_PATH%\bin` is on `PATH`, copy those DLLs next to the exe, or use the CPU-only build (`npm run build:cpu`).

**App won't start: `onnxruntime.dll` / `sherpa-onnx-c-api.dll` not found**
- In dev they're copied to `target\debug` by the build. For a release build, use `npm run tauri build`, which bundles them; don't run `cargo build` by itself.

**Build error: `Unable to find libclang`**
- Install LLVM and `setx LIBCLANG_PATH "C:\Program Files\LLVM\bin"`, then open a new terminal.

**Build error from nvcc / CMake**
- Open a fresh terminal after installing Build Tools and CUDA. CUDA 12.8+ supports VS 2022.
- Delete `target\` if a previous failed configure was cached.

**Captions lag or "Transcription is catching up"**
- The GPU is slower than real time (e.g. running on CPU). Use a smaller model (`large-v3-turbo-q5_0`, `small`, `base`), or check that the GPU is used: **Settings** shows the last run's device.

**Repeated or ghost phrases ("Thank you.")**
- Install `silero_vad.onnx`; the energy fallback VAD lets more noise through.
- Set **Language** explicitly (e.g. `en`) instead of `auto`.

**Everyone is "Speaker 1" / one person has several labels**
- Make sure the speaker model is installed (Settings shows ✓).
- Adjust **Same-speaker threshold**: higher = more speakers, lower = fewer.
- Use the rename table: renaming two labels to the same name merges them.

**My own voice appears twice (mic + desktop)**
- Use headphones, or keep **Drop mic echo** on.

**Summary failed**
- Settings shows whether the API key was loaded.
- 401 = bad key. 404 = wrong `CLAUDE_MODEL`.
- Overloads and timeouts are retried automatically. The transcript is always saved first.

**Discord failed**
- 401/404 = the webhook was deleted or the URL is wrong.
- Very large transcripts (> 9.5 MB) can't be attached; they remain on disk.

**Logs**
- `%APPDATA%\NoteTranscribe\logs\app.log`.

---

## Tests

The hardware-independent logic lives in the `meeting-core` crate and is unit-tested. The tests run on any OS and need no GPU or audio device:

```powershell
cargo test -p meeting-core     # or: npm test
```

Covered:
- merging the two streams by time;
- Discord message splitting (limits, boundaries, Unicode, code fences);
- prompt templating and chunking;
- settings load/save, including missing, empty, corrupt, partially invalid and BOM-prefixed files;
- mic labelling with "Mic is only me" on and off;
- rename handling;
- speaker clustering and refinement;
- resampling (rate, frequency preservation, anti-aliasing);
- wall-clock gap filling for silent loopback;
- the fallback VAD;
- the hallucination filter, echo suppression, and the Claude SSE stream parser.

---

## Project structure

```
crates/meeting-core/     Pure logic + tests (no audio/GPU/network)
  src/merge.rs           merge streams, render transcript, coalesce turns
  src/labels.rs          mic/desktop labelling rules, rename table
  src/discord_split.rs   2000-char message splitting
  src/prompt.rs          {{PLACEHOLDER}} templating, chunking for long meetings
  src/settings.rs        Settings struct, robust load/save
  src/cluster.rs         online speaker clustering + end-of-meeting refinement
  src/timeline.rs        keeps both streams aligned to wall-clock time
  src/resample.rs        downmix + streaming resampler to 16 kHz
  src/vad.rs             energy VAD fallback
  src/hallucination.rs   Whisper output cleanup
  src/echo.rs            mic echo suppression
  src/sse.rs             Server-Sent Events parser for the Claude stream
src-tauri/               Tauri app (Rust backend)
  src/audio.rs           WASAPI loopback + mic capture (cpal), device changes
  src/pipeline.rs        per-stream: resample, align, VAD, diarize, partial captions
  src/worker.rs          GPU job queue (finals before partials) + Whisper worker
  src/transcriber.rs     whisper.cpp wrapper, CUDA detection, CPU fallback
  src/vad.rs             Silero VAD (sherpa-onnx)
  src/diarize.rs         speaker embeddings (sherpa-onnx) + clustering
  src/session.rs         recording session, crash-safe transcript writer, finalize
  src/meeting.rs         finished meeting files (txt/json/summary)
  src/summarizer.rs      Claude Messages API: streaming, retries, chunking
  src/discord.rs         webhook posting, splitting, attachments, rate limits
  src/settings_store.rs  debounced + on-exit settings persistence
  src/config.rs          .env loading (secrets redacted from logs)
  src/commands.rs        Tauri commands used by the UI
src/                     Frontend (TypeScript, no framework)
prompts/summary_prompt.md   ← edit me
scripts/download-models.ps1 model downloader
scripts/copy-dlls.mjs       bundles sherpa-onnx DLLs into the installer
```
