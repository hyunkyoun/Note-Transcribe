//! Note Transcribe – passive meeting transcriber for Windows 11.

mod audio;
mod commands;
mod config;
mod diarize;
mod discord;
mod events;
mod logging;
mod meeting;
mod paths;
mod pipeline;
mod session;
mod settings_store;
mod state;
mod summarizer;
mod transcriber;
mod vad;
mod worker;

use tauri::{Manager, RunEvent, WindowEvent};

use crate::settings_store::SettingsStore;
use crate::state::AppState;

fn save_and_cleanup(app: &tauri::AppHandle) {
    let state = app.state::<AppState>();
    state.settings.save_now();
    // If the window is closed mid-recording, finish quickly and keep everything captured so far.
    let rec = state.recording.lock().take();
    if let Some(rec) = rec {
        log::info!("app closing during recording; saving transcript");
        let meeting = rec.stop(true);
        *state.meeting.lock() = Some(meeting);
    }
}

pub fn run() {
    logging::init();
    log::info!("Note Transcribe {} starting", env!("CARGO_PKG_VERSION"));

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            if let Ok(dir) = app.path().resource_dir() {
                paths::set_resource_dir(dir);
            }
            config::load_env();
            let store = SettingsStore::load(paths::settings_path());
            app.manage(AppState::new(store));
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { .. } = event {
                save_and_cleanup(window.app_handle());
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_settings,
            commands::update_settings,
            commands::list_devices,
            commands::get_status,
            commands::reload_env,
            commands::start_recording,
            commands::stop_recording,
            commands::get_meeting,
            commands::get_transcript,
            commands::set_renames,
            commands::summarize_and_send,
            commands::send_to_discord,
            commands::open_location,
        ])
        .build(tauri::generate_context!())
        .expect("error while building the Tauri application");

    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            save_and_cleanup(handle);
        }
    });
}
