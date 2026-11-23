This folder is filled by scripts/copy-dlls.mjs during `npm run tauri build`
(Tauri's beforeBundleCommand). It receives the sherpa-onnx runtime DLLs
(sherpa-onnx-c-api.dll, onnxruntime.dll, ...) so the installer ships them next
to the exe. The DLLs are git-ignored; this README keeps the folder non-empty.
