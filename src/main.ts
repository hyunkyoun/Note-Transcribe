import { api, on, type AppStatus, type CaptionEvent, type DeviceList, type MeetingView, type Settings, type Source } from "./api";

// ---------- helpers ----------
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const el = <K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};
const fmtTime = (ms: number) => {
  const s = Math.floor(ms / 1000);
  return [Math.floor(s / 3600), Math.floor(s / 60) % 60, s % 60].map((n) => String(n).padStart(2, "0")).join(":");
};
const errText = (e: unknown) => (typeof e === "string" ? e : e instanceof Error ? e.message : JSON.stringify(e));
function debounce<A extends unknown[]>(fn: (...a: A) => void, ms: number) {
  let t: number | undefined;
  return (...a: A) => {
    clearTimeout(t);
    t = window.setTimeout(() => fn(...a), ms);
  };
}
/** Stable, distinct color per speaker label ("Speaker 1"/"Speaker 2" hash to neighbouring slots). */
const PALETTE = ["#5ea3ff", "#4cc38a", "#f2b84b", "#e879a6", "#a78bfa", "#f97316", "#22d3ee", "#a3e635"];
function speakerColor(label: string): string {
  let h = 0;
  for (const c of label) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return PALETTE[h % PALETTE.length];
}

// ---------- state ----------
let settings: Settings;
let status: AppStatus | null = null;
let devices: DeviceList = { outputs: [], inputs: [] };
let recording = false;
let recordStart = 0;
let meeting: MeetingView | null = null;
let autoScroll = true;

// ---------- tabs ----------
document.querySelectorAll<HTMLButtonElement>(".tab").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === b));
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${b.dataset.view}`));
    if (b.dataset.view === "settings") refreshStatus();
  }),
);

// ---------- notices ----------
function notice(message: string, level: "info" | "warn" | "error" = "info", source?: Source | null) {
  const box = $("notices");
  const n = el("div", `notice ${level}`);
  const prefix = source === "mic" ? "Microphone: " : source === "desktop" ? "Desktop audio: " : "";
  n.textContent = prefix + message;
  const close = el("button", "x", "×");
  close.onclick = () => n.remove();
  n.append(close);
  box.prepend(n);
  while (box.children.length > 5) box.lastElementChild?.remove();
  if (level === "info") setTimeout(() => n.remove(), 8000);
}

// ---------- settings ----------
const saveSettings = debounce(async () => {
  try {
    settings = await api.updateSettings(settings);
    applyDisplaySettings();
  } catch (e) {
    notice(`Could not save settings: ${errText(e)}`, "error");
  }
}, 250);

function setSetting<K extends keyof Settings>(key: K, value: Settings[K]) {
  settings[key] = value;
  syncSettingsInputs();
  saveSettings();
}

function applyDisplaySettings() {
  document.documentElement.style.setProperty("--caption-size", `${settings.caption_font_size}px`);
  $("thr-val").textContent = settings.diarization_threshold.toFixed(2);
  $("font-val").textContent = `${settings.caption_font_size}px`;
  $("mic-mode-hint").textContent = settings.mic_is_only_me
    ? `(labelled "${settings.my_name.trim() || "Me"}")`
    : "(shared mic: speakers separated)";
  $<HTMLButtonElement>("summarize").textContent = settings.post_to_discord ? "Summarize & send to Discord" : "Summarize";
}

function syncSettingsInputs() {
  document.querySelectorAll<HTMLInputElement | HTMLTextAreaElement>("[data-setting]").forEach((input) => {
    const key = input.dataset.setting as keyof Settings;
    const v = settings[key];
    if (input instanceof HTMLInputElement && input.type === "checkbox") input.checked = Boolean(v);
    else if (document.activeElement !== input) input.value = v === null ? "" : String(v);
  });
  $<HTMLInputElement>("live-capture-desktop").checked = settings.capture_desktop;
  $<HTMLInputElement>("live-capture-mic").checked = settings.capture_mic;
  renderDeviceSelects();
  applyDisplaySettings();
}

document.querySelectorAll<HTMLInputElement | HTMLTextAreaElement>("[data-setting]").forEach((input) => {
  const key = input.dataset.setting as keyof Settings;
  const handler = () => {
    let value: unknown;
    if (input instanceof HTMLInputElement && input.type === "checkbox") value = input.checked;
    else if (input.dataset.number !== undefined) value = Number(input.value);
    else value = input.value;
    (settings as unknown as Record<string, unknown>)[key] = value;
    applyDisplaySettings();
    if (input instanceof HTMLInputElement && input.type === "checkbox") syncSettingsInputs();
    saveSettings();
  };
  input.addEventListener(input instanceof HTMLInputElement && input.type === "checkbox" ? "change" : "input", handler);
});

$<HTMLInputElement>("live-capture-desktop").addEventListener("change", (e) => setSetting("capture_desktop", (e.target as HTMLInputElement).checked));
$<HTMLInputElement>("live-capture-mic").addEventListener("change", (e) => setSetting("capture_mic", (e.target as HTMLInputElement).checked));

// ---------- devices ----------
const DEFAULT_VALUE = "__default__";

function fillSelect(sel: HTMLSelectElement, list: { name: string; is_default?: boolean }[], current: string | null, kind: string) {
  sel.innerHTML = "";
  const def = list.find((d) => d.is_default);
  sel.append(new Option(`System default${def ? ` (${def.name})` : ""} — follows Windows`, DEFAULT_VALUE));
  for (const d of list) sel.append(new Option(d.name, d.name));
  if (current && !list.some((d) => d.name === current)) {
    sel.append(new Option(`${current} (not connected)`, current));
  }
  sel.value = current ?? DEFAULT_VALUE;
  sel.title = `${kind} device`;
}

function renderDeviceSelects() {
  if (!settings) return;
  for (const id of ["live-output-device", "set-output-device"]) fillSelect($(id), devices.outputs, settings.output_device, "Output");
  for (const id of ["live-mic-device", "set-mic-device"]) fillSelect($(id), devices.inputs, settings.mic_device, "Microphone");
  for (const id of ["live-output-device", "set-output-device", "live-mic-device", "set-mic-device"]) {
    $<HTMLSelectElement>(id).disabled = recording;
  }
}

for (const id of ["live-output-device", "set-output-device"]) {
  $<HTMLSelectElement>(id).addEventListener("change", (e) => {
    const v = (e.target as HTMLSelectElement).value;
    setSetting("output_device", v === DEFAULT_VALUE ? null : v);
  });
}
for (const id of ["live-mic-device", "set-mic-device"]) {
  $<HTMLSelectElement>(id).addEventListener("change", (e) => {
    const v = (e.target as HTMLSelectElement).value;
    setSetting("mic_device", v === DEFAULT_VALUE ? null : v);
  });
}

async function loadDevices() {
  try {
    devices = await api.listDevices();
    renderDeviceSelects();
  } catch (e) {
    notice(`Could not list audio devices: ${errText(e)}`, "error");
  }
}
$("refresh-devices").addEventListener("click", loadDevices);

// ---------- status / paths ----------
async function refreshStatus() {
  try {
    status = await api.getStatus();
  } catch {
    return;
  }
  const s = status;
  const models = $("model-list");
  models.innerHTML = "";
  for (const m of ["large-v3-turbo", "large-v3-turbo-q5_0", "large-v3", "medium", "small", "small.en", "base", "base.en", "tiny"]) {
    const o = new Option(m + (s.whisperModels.includes(m) ? " ✓ installed" : ""), m);
    models.append(o);
  }
  for (const m of s.whisperModels) if (![...models.querySelectorAll("option")].some((o) => o.value === m)) models.append(new Option(`${m} ✓ installed`, m));

  const info: string[] = [];
  info.push(s.whisperModels.length ? `Installed models: ${s.whisperModels.join(", ")}` : "⚠ No Whisper models found in the models folder.");
  info.push(s.hasVadModel ? "Silero VAD ✓" : "⚠ silero_vad.onnx missing (basic VAD will be used)");
  info.push(s.speakerModel ? `Speaker model ✓ (${s.speakerModel})` : "⚠ No speaker-embedding model (no diarization)");
  info.push(s.cudaBuild ? "CUDA build" : "CPU-only build");
  if (s.engine) {
    info.push(`Last run: ${s.engine.model} on ${s.engine.gpu ? s.engine.gpu_name ?? "GPU" : "CPU"}` + (s.engine.vram_used_mb ? ` (~${s.engine.vram_used_mb} MB VRAM)` : ""));
  }
  $("model-info").innerHTML = info.map((l) => `<div>${escapeHtml(l)}</div>`).join("");

  const c = s.config;
  const rows: [string, string][] = [
    ["Settings file", s.settingsPath],
    ["Data folder", s.dataDir],
    ["Models folder", s.modelsDir],
    ["Transcripts", s.transcriptsDir],
    ["Summary prompt", s.promptPath ?? "⚠ prompts/summary_prompt.md not found (built-in default will be used)"],
    [".env", c.env_path ?? "⚠ not found — copy .env.example to .env"],
    ["Anthropic API key", c.has_anthropic_key ? `✓ set (model: ${c.claude_model})` : "✗ not set (ANTHROPIC_API_KEY)"],
    ["Discord webhook", c.has_discord_webhook ? "✓ set" : "✗ not set (DISCORD_WEBHOOK_URL)"],
  ];
  $("paths").innerHTML = rows.map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd>`).join("");
  if (s.settingsWarning) notice(s.settingsWarning, "warn");
}

function escapeHtml(s: string) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}

document.querySelectorAll<HTMLButtonElement>("[data-open]").forEach((b) =>
  b.addEventListener("click", async () => {
    try {
      await api.open(b.dataset.open!);
    } catch (e) {
      notice(errText(e), "warn");
    }
  }),
);
$("reload-env").addEventListener("click", async () => {
  const c = await api.reloadEnv();
  notice(
    `.env reloaded: Anthropic key ${c.has_anthropic_key ? "✓" : "✗"}, Discord webhook ${c.has_discord_webhook ? "✓" : "✗"}, model ${c.claude_model}`,
    c.has_anthropic_key ? "info" : "warn",
  );
  refreshStatus();
});

// ---------- recording ----------
const startStop = $<HTMLButtonElement>("start-stop");

function setRecordingUi(state: "idle" | "recording" | "finalizing") {
  recording = state === "recording";
  const ind = $("rec-indicator");
  ind.className = `rec-indicator ${state}`;
  $("rec-label").textContent = state === "recording" ? "Recording" : state === "finalizing" ? "Finishing…" : "Idle";
  startStop.textContent = state === "recording" ? "■ Stop" : state === "finalizing" ? "Finishing…" : "● Start";
  startStop.classList.toggle("danger", state === "recording");
  startStop.disabled = state === "finalizing";
  for (const id of ["live-capture-desktop", "live-capture-mic"]) $<HTMLInputElement>(id).disabled = state !== "idle";
  renderDeviceSelects();
}

startStop.addEventListener("click", async () => {
  if (recording) {
    setRecordingUi("finalizing");
    try {
      meeting = await api.stop();
      renderMeeting();
      notice(`Saved ${meeting.segmentCount} segments to ${meeting.transcriptPath}`, "info");
    } catch (e) {
      notice(`Stop failed: ${errText(e)}`, "error");
    }
    setRecordingUi("idle");
    $("partials").innerHTML = "";
    return;
  }
  if (!settings.capture_desktop && !settings.capture_mic) {
    notice("Turn on desktop audio and/or the microphone first.", "warn");
    return;
  }
  startStop.disabled = true;
  try {
    await api.start();
    $("captions").innerHTML = "";
    $("partials").innerHTML = "";
    $("meeting-panel").classList.add("hidden");
    $("progress-log").innerHTML = "";
    recordStart = Date.now();
    autoScroll = true;
    setRecordingUi("recording");
  } catch (e) {
    notice(errText(e), "error");
    setRecordingUi("idle");
  } finally {
    startStop.disabled = false;
  }
});

setInterval(async () => {
  if (!recording) return;
  $("rec-timer").textContent = fmtTime(Date.now() - recordStart);
}, 500);
setInterval(async () => {
  if (!recording) return;
  try {
    const s = await api.getStatus();
    if (s.elapsedMs !== null) recordStart = Date.now() - s.elapsedMs; // stay in sync with the backend clock
    $("queue-status").textContent = s.pendingJobs > 2 ? `Transcription is catching up (${s.pendingJobs} segments queued)…` : "";
  } catch {
    /* ignore */
  }
}, 3000);

// ---------- captions ----------
const captionsEl = $("captions");
const wrap = captionsEl.parentElement!;
wrap.addEventListener("scroll", () => {
  autoScroll = wrap.scrollTop + wrap.clientHeight >= wrap.scrollHeight - 40;
  $("jump-latest").classList.toggle("hidden", autoScroll);
});
$("jump-latest").addEventListener("click", () => {
  autoScroll = true;
  scrollToEnd();
});
function scrollToEnd() {
  if (autoScroll) wrap.scrollTop = wrap.scrollHeight;
}

function captionLine(e: CaptionEvent, partial: boolean) {
  const line = el("div", `cap ${e.source}${partial ? " partial" : ""}`);
  line.append(el("span", "ts", fmtTime(e.startMs)));
  const sp = el("span", "spk", partial ? `${e.speaker}…` : e.speaker);
  sp.style.color = speakerColor(e.speaker);
  line.append(sp, el("span", "txt", e.text));
  return line;
}

function insertFinal(e: CaptionEvent) {
  const line = captionLine(e, false);
  line.dataset.start = String(e.startMs);
  // Keep lines ordered by start time (the two streams finish out of order).
  const children = [...captionsEl.children] as HTMLElement[];
  const after = children.reverse().find((c) => Number(c.dataset.start) <= e.startMs);
  if (after) after.after(line);
  else captionsEl.prepend(line);
}

on.caption((e) => {
  const partials = $("partials");
  const existing = partials.querySelector<HTMLElement>(`[data-source="${e.source}"]`);
  if (e.kind === "partial") {
    const line = captionLine(e, true);
    line.dataset.source = e.source;
    if (existing) existing.replaceWith(line);
    else partials.append(line);
  } else {
    existing?.remove();
    if (e.kind === "final") insertFinal(e);
  }
  scrollToEnd();
});

on.status((e) => notice(e.message, e.level, e.source));
on.engine((e) => {
  const s = $("engine-status");
  s.textContent = e.message + (e.info?.note ? ` — ${e.info.note}` : "");
  s.className = e.state === "error" ? "error" : "muted";
  if (e.state === "error") notice(e.message, "error");
  if (e.info?.note) notice(e.info.note, "warn");
});
on.recordingState((s) => setRecordingUi(s as "idle" | "recording" | "finalizing"));

// ---------- meeting panel ----------
function renderMeeting() {
  if (!meeting) return;
  $("meeting-panel").classList.remove("hidden");
  $("meeting-title").textContent = `${meeting.startedLabel} · ${meeting.duration} · ${meeting.segmentCount} segments`;
  const rows = $("rename-rows");
  rows.innerHTML = "";
  for (const sp of meeting.speakers) {
    const tr = el("tr");
    const label = el("td");
    const chip = el("span", "spk", sp);
    chip.style.color = speakerColor(sp);
    label.append(chip);
    const td = el("td");
    const input = el("input") as HTMLInputElement;
    input.type = "text";
    input.placeholder = sp;
    input.value = meeting.renames[sp] ?? "";
    input.dataset.label = sp;
    td.append(input);
    tr.append(label, td);
    rows.append(tr);
  }
  if (!meeting.speakers.length) rows.innerHTML = `<tr><td colspan="2" class="muted">No speech was transcribed.</td></tr>`;
  const box = $("summary-box");
  box.classList.toggle("hidden", !meeting.summary);
  $("summary-text").textContent = meeting.summary ?? "";
  if (meeting.summary) (box as HTMLDetailsElement).open = true;
}

function collectRenames() {
  const r: Record<string, string> = {};
  document.querySelectorAll<HTMLInputElement>("#rename-rows input").forEach((i) => {
    if (i.value.trim()) r[i.dataset.label!] = i.value.trim();
  });
  return r;
}

function progress(message: string, ok: boolean | null) {
  const li = el("li", ok === true ? "ok" : ok === false ? "fail" : "", message);
  $("progress-log").append(li);
}

async function busy(btn: HTMLButtonElement, fn: () => Promise<void>) {
  const buttons = ["summarize", "save-names", "send-transcript", "resend"].map((id) => $<HTMLButtonElement>(id));
  buttons.forEach((b) => (b.disabled = true));
  const old = btn.textContent;
  btn.textContent = "Working…";
  try {
    await fn();
  } finally {
    btn.textContent = old;
    buttons.forEach((b) => (b.disabled = false));
    applyDisplaySettings();
  }
}

$("save-names").addEventListener("click", (e) =>
  busy(e.target as HTMLButtonElement, async () => {
    try {
      meeting = await api.setRenames(collectRenames());
      renderMeeting();
      progress("Names saved to the transcript.", true);
    } catch (err) {
      progress(errText(err), false);
    }
  }),
);
$("summarize").addEventListener("click", (e) =>
  busy(e.target as HTMLButtonElement, async () => {
    $("progress-log").innerHTML = "";
    try {
      meeting = await api.summarize(collectRenames());
      renderMeeting();
    } catch (err) {
      // The backend already reported details through progress events.
      if (!$("progress-log").textContent?.includes(errText(err))) progress(errText(err), false);
    }
  }),
);
$("send-transcript").addEventListener("click", (e) =>
  busy(e.target as HTMLButtonElement, async () => {
    try {
      await api.setRenames(collectRenames());
      meeting = await api.sendToDiscord(true);
      renderMeeting();
    } catch (err) {
      progress(errText(err), false);
    }
  }),
);
$("resend").addEventListener("click", (e) =>
  busy(e.target as HTMLButtonElement, async () => {
    try {
      meeting = await api.sendToDiscord(false);
      renderMeeting();
    } catch (err) {
      progress(errText(err), false);
    }
  }),
);
$("open-transcript").addEventListener("click", () => api.open("transcript").catch((e) => notice(errText(e), "warn")));

on.progress((e) => {
  if (e.stage === "summarizing" || e.stage === "retrying") {
    const last = $("progress-log").lastElementChild;
    if (last && last.classList.contains("live")) last.remove();
    const li = el("li", "live", e.message);
    $("progress-log").append(li);
    return;
  }
  $("progress-log").querySelector(".live")?.remove();
  progress(e.message, e.ok);
});

// ---------- boot ----------
async function boot() {
  settings = await api.getSettings();
  syncSettingsInputs();
  await loadDevices();
  await refreshStatus();
  const st = status;
  if (st?.recording) {
    recordStart = Date.now() - (st.elapsedMs ?? 0);
    setRecordingUi("recording");
  } else {
    setRecordingUi("idle");
  }
  meeting = await api.getMeeting();
  if (meeting) renderMeeting();
  if (st && !st.whisperModels.length) notice(`No Whisper model found in ${st.modelsDir}. See README → "Download the models".`, "warn");
  if (st && !st.config.has_anthropic_key) notice("ANTHROPIC_API_KEY is not set in .env — summaries are disabled until you add it.", "warn");
}

boot().catch((e) => notice(`Startup failed: ${errText(e)}`, "error"));
