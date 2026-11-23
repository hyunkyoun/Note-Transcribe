// Typed wrappers around the Rust commands and events.
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

export type Source = "mic" | "desktop";
export type Level = "info" | "warn" | "error";

/** Mirrors meeting_core::settings::Settings (snake_case on the wire). */
export interface Settings {
  my_name: string;
  mic_is_only_me: boolean;
  capture_desktop: boolean;
  capture_mic: boolean;
  output_device: string | null;
  mic_device: string | null;
  whisper_model: string;
  fallback_model: string;
  use_gpu: boolean;
  language: string;
  initial_prompt: string;
  diarization_threshold: number;
  max_speakers: number;
  suppress_mic_echo: boolean;
  save_raw_audio: boolean;
  caption_font_size: number;
  auto_summarize_on_stop: boolean;
  post_to_discord: boolean;
  keep_model_loaded: boolean;
  models_dir: string;
  transcripts_dir: string;
}

export interface DeviceInfo { name: string; is_default: boolean }
export interface DeviceList { outputs: DeviceInfo[]; inputs: DeviceInfo[] }

export interface EngineInfo {
  model: string;
  path: string;
  gpu: boolean;
  gpu_name: string | null;
  vram_used_mb: number | null;
  note: string | null;
}

export interface ConfigStatus {
  has_anthropic_key: boolean;
  claude_model: string;
  has_discord_webhook: boolean;
  env_path: string | null;
}

export interface AppStatus {
  recording: boolean;
  sessionId: string | null;
  elapsedMs: number | null;
  pendingJobs: number;
  busy: boolean;
  config: ConfigStatus;
  promptPath: string | null;
  modelsDir: string;
  whisperModels: string[];
  hasVadModel: boolean;
  speakerModel: string | null;
  dataDir: string;
  settingsPath: string;
  transcriptsDir: string;
  cudaBuild: boolean;
  engine: EngineInfo | null;
  settingsWarning: string | null;
}

export type Renames = Record<string, string>;

export interface MeetingView {
  id: string;
  startedLabel: string;
  duration: string;
  speakers: string[];
  renames: Renames;
  segmentCount: number;
  transcriptPath: string;
  summary: string | null;
  summaryPath: string | null;
  discordPosted: boolean;
}

export interface CaptionEvent {
  kind: "partial" | "final" | "clear";
  source: Source;
  id: number | null;
  startMs: number;
  endMs: number;
  speaker: string;
  text: string;
}

export interface StatusEvent { source: Source | null; level: Level; message: string }
export interface EngineEvent { state: "loading" | "ready" | "error" | "idle"; message: string; info: EngineInfo | null }
export interface ProgressEvent { stage: string; message: string; ok: boolean | null }

export const api = {
  getSettings: () => invoke<Settings>("get_settings"),
  updateSettings: (settings: Settings) => invoke<Settings>("update_settings", { settings }),
  listDevices: () => invoke<DeviceList>("list_devices"),
  getStatus: () => invoke<AppStatus>("get_status"),
  reloadEnv: () => invoke<ConfigStatus>("reload_env"),
  start: () => invoke<{ sessionId: string; transcriptsDir: string }>("start_recording"),
  stop: () => invoke<MeetingView>("stop_recording"),
  getMeeting: () => invoke<MeetingView | null>("get_meeting"),
  setRenames: (renames: Renames) => invoke<MeetingView>("set_renames", { renames }),
  summarize: (renames: Renames) => invoke<MeetingView>("summarize_and_send", { renames }),
  sendToDiscord: (transcriptOnly: boolean) => invoke<MeetingView>("send_to_discord", { transcriptOnly }),
  open: (which: string) => invoke<void>("open_location", { which }),
};

export const on = {
  caption: (cb: (e: CaptionEvent) => void): Promise<UnlistenFn> => listen<CaptionEvent>("caption", (e) => cb(e.payload)),
  status: (cb: (e: StatusEvent) => void) => listen<StatusEvent>("status", (e) => cb(e.payload)),
  engine: (cb: (e: EngineEvent) => void) => listen<EngineEvent>("engine", (e) => cb(e.payload)),
  progress: (cb: (e: ProgressEvent) => void) => listen<ProgressEvent>("progress", (e) => cb(e.payload)),
  recordingState: (cb: (s: string) => void) => listen<string>("recording-state", (e) => cb(e.payload)),
};
