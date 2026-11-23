// Copies the sherpa-onnx runtime DLLs produced by the Rust build into
// src-tauri/libs so `tauri build` bundles them next to the exe.
// Runs automatically as Tauri's beforeBundleCommand.
import { readdirSync, copyFileSync, existsSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dest = join(root, "src-tauri", "libs");
mkdirSync(dest, { recursive: true });

const candidates = [join(root, "target", "release"), join(root, "target", "x86_64-pc-windows-msvc", "release")];
let copied = 0;
for (const dir of candidates) {
  if (!existsSync(dir)) continue;
  for (const f of readdirSync(dir)) {
    if (f.toLowerCase().endsWith(".dll")) {
      copyFileSync(join(dir, f), join(dest, f));
      console.log(`copied ${f}`);
      copied++;
    }
  }
}
if (copied === 0) {
  console.warn("copy-dlls: no DLLs found in target/release (sherpa-onnx may be linked statically).");
}
