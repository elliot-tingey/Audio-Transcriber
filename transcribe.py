#!/usr/bin/env python3
import json
import time
import threading
import queue
from pathlib import Path
from datetime import datetime

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

from faster_whisper import WhisperModel

# Supported audio extensions
SUPPORTED_EXTS = {".mp3", ".wav", ".m4a", ".mp4", ".aac", ".flac", ".ogg"}

# Default output directory = Downloads
DEFAULT_DOWNLOADS_DIR = Path.home() / "Downloads"

# Simple config file in the user's home directory
CONFIG_PATH = Path.home() / ".whisper_call_transcriber_config.json"


def load_output_dir() -> Path:
    """
    Load the last-used output directory from config.
    Falls back to Downloads if missing/invalid.
    """
    try:
        if CONFIG_PATH.exists():
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            out = Path(data.get("output_dir", ""))
            if out.exists() and out.is_dir():
                return out
    except Exception:
        # If anything goes weird, ignore and fall back.
        pass

    # Ensure Downloads exists
    DEFAULT_DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_DOWNLOADS_DIR


def save_output_dir(output_dir: Path):
    """Save the chosen output directory to config."""
    try:
        CONFIG_PATH.write_text(
            json.dumps({"output_dir": str(output_dir)}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        # Not fatal if this fails; just ignore.
        pass


def load_model():
    """
    Load the Whisper 'small' model optimized for minimal hardware:
    - CPU device
    - int8 quantization
    """
    device = "cpu"
    compute_type = "int8"  # Fast / low RAM on CPU

    model = WhisperModel(
        "small",
        device=device,
        compute_type=compute_type,
        cpu_threads=0,  # 0 = auto / use all cores
    )
    return model


class TranscriberApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Call Transcriber")
        self.root.geometry("650x450")
        self.root.resizable(False, False)

        # Message queue from worker thread -> GUI thread
        self.queue = queue.Queue()

        # File lists
        self.audio_files: list[Path] = []   # files the user has staged
        self.worker_files: list[Path] = []  # snapshot used by the worker
        self.total_files = 0
        self.files_done = 0  # how many files fully finished

        # Output directory (default: Downloads or last-used)
        self.output_dir: Path = load_output_dir()

        # Transcribing state
        self.is_transcribing = False

        # UI elements
        self.create_widgets()

        # Start polling the queue
        self.root.after(100, self.process_queue)

    def create_widgets(self):
        padding = {"padx": 10, "pady": 5}

        self.info_label = ttk.Label(
            self.root,
            text="Select call recording files to transcribe.",
            wraplength=620,
            justify="center",
        )
        self.info_label.pack(padx=10, pady=6)

               # --- File progress (percent within current file) ---
        self.file_progress_label = ttk.Label(
            self.root,
            text="File progress: 0%",
        )
        self.info_label.pack(padx=10, pady=6)

        self.file_progress_bar = ttk.Progressbar(
            self.root,
            orient="horizontal",
            length=620,
            mode="determinate",
            maximum=100,
        )
        self.info_label.pack(padx=10, pady=6)

        # --- Overall progress (files done out of total) ---
        self.overall_progress_label = ttk.Label(
            self.root,
            text="Overall: 0 / 0 files",
        )
        self.info_label.pack(padx=10, pady=6)

        self.overall_progress_bar = ttk.Progressbar(
            self.root,
            orient="horizontal",
            length=620,
            mode="determinate",
            maximum=1,   # will be set when a run starts
        )
        self.info_label.pack(padx=10, pady=6)


        self.current_text_label = ttk.Label(
            self.root,
            text="Current text: (none)",
            wraplength=620,
            justify="center",
        )
        self.info_label.pack(padx=10, pady=6)

        self.current_file_label = ttk.Label(
            self.root,
            text="Current file: None",
            wraplength=620,
            justify="center",
        )
        self.info_label.pack(padx=10, pady=6)

        # Selected files list
        files_frame = ttk.LabelFrame(self.root, text="Selected files")
        files_frame.pack(fill="both", expand=False, padx=10, pady=5)

        self.file_listbox = tk.Listbox(
            files_frame,
            height=8,
            width=80,
            activestyle="none",
        )
        self.file_listbox.pack(side="left", fill="both", expand=True, padx=(5, 0), pady=5)

        scrollbar = ttk.Scrollbar(
            files_frame,
            orient="vertical",
            command=self.file_listbox.yview,
        )
        scrollbar.pack(side="right", fill="y", padx=(0, 5), pady=5)
        self.file_listbox.config(yscrollcommand=scrollbar.set)

        # Output directory chooser
        out_frame = ttk.Frame(self.root)
        out_frame.pack(fill="x", padx=10, pady=(5, 10))

        self.output_label = ttk.Label(
            out_frame,
            text=f"Output folder: {self.output_dir}",
            wraplength=500,
            justify="left",
        )
        self.output_label.pack(side="left", padx=(0, 10))

        self.change_output_button = ttk.Button(
            out_frame,
            text="Change Output Folder...",
            command=self.on_change_output_dir,
        )
        self.change_output_button.pack(side="left")

        # Buttons: Select / Start / Quit
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(pady=10)

        self.select_button = ttk.Button(
            self.button_frame,
            text="Select Files",
            command=self.on_select_files,
        )
        self.select_button.pack(side="left", padx=5)

        self.start_button = ttk.Button(
            self.button_frame,
            text="Start",
            command=self.on_start_transcription,
            state="disabled",  # disabled until files selected
        )
        self.start_button.pack(side="left", padx=5)

        self.quit_button = ttk.Button(
            self.button_frame,
            text="Quit",
            command=self.root.destroy,
        )
        self.quit_button.pack(side="left", padx=5)

    # ---------- UI actions ----------

    def on_change_output_dir(self):
        """Let the user choose where to save transcripts."""
        new_dir = filedialog.askdirectory(
            title="Select output folder for transcripts",
            initialdir=str(self.output_dir),
        )
        if not new_dir:
            return

        out_path = Path(new_dir)
        try:
            out_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Could not create / access directory:\n{new_dir}\n\n{e}",
            )
            return

        self.output_dir = out_path
        self.output_label.config(text=f"Output folder: {self.output_dir}")
        save_output_dir(self.output_dir)

    def on_select_files(self):
        """Select additional files and append to the list (no duplicates)."""
        if self.is_transcribing:
            # Just to be safe; shouldn't happen if buttons are managed correctly.
            messagebox.showinfo(
                "Busy",
                "Transcription is currently in progress. Please wait.",
            )
            return

        filepaths = filedialog.askopenfilenames(
            title="Select call recording(s) to transcribe",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.m4a *.flac *.ogg *.aac *.mp4"),
                ("All files", "*.*"),
            ],
        )

        if not filepaths:
            return

        added = 0
        for p in filepaths:
            path_obj = Path(p)
            if path_obj.suffix.lower() not in SUPPORTED_EXTS:
                continue

            # Avoid duplicates: compare by resolved path if possible
            try:
                resolved = path_obj.resolve()
            except Exception:
                resolved = path_obj

            already = any(
                (f.resolve() if f.exists() else f) == resolved
                for f in self.audio_files
            )
            if not already:
                self.audio_files.append(path_obj)
                self.file_listbox.insert(tk.END, path_obj.name)
                added += 1

        self.total_files = len(self.audio_files)
        if self.total_files == 0:
            messagebox.showwarning(
                "No supported files",
                f"No supported audio files selected.\n\n"
                f"Supported: {', '.join(sorted(SUPPORTED_EXTS))}",
            )
            self.start_button.config(state="disabled")
            return

        # Enable Start if we have at least one file
        self.start_button.config(state="normal")

        if added > 0:
            self.info_label.config(
                text=f"{self.total_files} file(s) selected. Ready to start."
            )
        else:
            self.info_label.config(
                text="No new files were added (all were already selected)."
            )

    def on_start_transcription(self):
        """Kick off transcription in a background thread."""
        if self.is_transcribing:
            return

        if not self.audio_files:
            messagebox.showwarning(
                "No files",
                "Please select one or more audio files first.",
            )
            return

        # Take a snapshot of current files for this run
        self.worker_files = list(self.audio_files)
        self.total_files = len(self.worker_files)
        self.files_done = 0
        self.is_transcribing = True

        # Clear staged files for the next run
        self.audio_files.clear()
        self.file_listbox.delete(0, tk.END)

        # Reset progress
        self.file_progress_bar["value"] = 0
        self.file_progress_label.config(text="File progress: 0%")

        self.overall_progress_bar["maximum"] = max(self.total_files, 1)
        self.overall_progress_bar["value"] = 0
        self.overall_progress_label.config(
            text=f"Overall: {self.files_done} / {self.total_files} files"
        )

        self.current_file_label.config(text="Current file: Preparing...")
        self.current_text_label.config(text="Current text: (none)")

        # Disable buttons while working
        self.select_button.config(state="disabled")
        self.start_button.config(state="disabled")
        self.change_output_button.config(state="disabled")

        # Start worker thread
        worker_thread = threading.Thread(
            target=self.worker_transcribe_all,
            daemon=True,
        )
        worker_thread.start()


    # ---------- Worker & queue ----------

    def worker_transcribe_all(self):
        try:
            start_time = time.time()
            self.queue.put(("status", "Loading model (small, int8 on CPU)..."))

            model = load_model()
            self.queue.put(("status", "Model loaded. Starting transcription..."))

            for idx, audio_path in enumerate(self.worker_files, start=1):
                file_start = time.time()
                self.queue.put((
                    "file_start",
                    idx,
                    self.total_files,
                    audio_path.name,
                ))

                # Transcribe this file, streaming segments
                segments, info = model.transcribe(
                    str(audio_path),
                    beam_size=1,        # greedy decoding, fastest
                    vad_filter=True,   # no extra VAD overhead
                    word_timestamps=False,
                )

                duration = getattr(info, "duration", None) or 0.0
                lines = []

                for seg in segments:
                    text = (seg.text or "").strip()
                    if not text:
                        continue
                    lines.append(text)

                    # Compute per-file progress based on segment end vs total duration
                    if duration > 0 and seg.end is not None:
                        percent = int(min(max(seg.end / duration * 100, 0), 100))
                    else:
                        percent = 0

                    # Update GUI with current text chunk and percent
                    self.queue.put(("segment", text, percent))

                # Timestamp when transcription finished
                timestamp_str = datetime.now().strftime("%H%M%S")
                out_name = f"Call{idx}_{timestamp_str}.txt"
                out_path = self.output_dir / out_name

                self.output_dir.mkdir(parents=True, exist_ok=True)

                # Write transcript with a blank line between each line
                with out_path.open("w", encoding="utf-8") as f:
                    for line in lines:
                        f.write(line + "\n\n")

                file_elapsed = time.time() - file_start
                self.queue.put((
                    "file_done",
                    idx,
                    self.total_files,
                    audio_path.name,
                    str(out_path),
                    file_elapsed,
                ))

            total_elapsed = time.time() - start_time
            self.queue.put(("all_done", total_elapsed))

        except Exception as e:
            self.queue.put(("error", str(e)))

    def process_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                self.handle_message(msg)
        except queue.Empty:
            pass

        # Keep polling
        self.root.after(100, self.process_queue)

    def handle_message(self, msg):
        msg_type = msg[0]

        if msg_type == "status":
            _, text = msg
            self.info_label.config(text=text)

        elif msg_type == "file_start":
            _, idx, total, filename = msg
            self.current_file_label.config(
                text=f"Current file: ({idx} / {total}) {filename}"
            )
            self.current_text_label.config(text="Current text: (none)")
            self.file_progress_bar["value"] = 0
            self.file_progress_label.config(text="File progress: 0%")

            self.overall_progress_label.config(
                text=f"Overall: {self.files_done} / {self.total_files} files"
            )
            self.overall_progress_bar["value"] = self.files_done


        elif msg_type == "segment":
            _, text, percent = msg
            # Show latest segment being transcribed
            self.current_text_label.config(
                text=f"Current text: {text}"
            )
            # Percent is per-file; X / N is files done
            self.file_progress_bar["value"] = percent
            self.file_progress_label.config(text=f"File progress: {percent}%")


        elif msg_type == "file_done":
            _, idx, total, filename, out_path, file_elapsed = msg

            # Mark this file as done
            self.files_done += 1

            # Force progress to 100% for this file
            self.file_progress_bar["value"] = 100
            self.file_progress_label.config(text="File progress: 100%")

            self.overall_progress_bar["value"] = self.files_done
            self.overall_progress_label.config(
                text=f"Overall: {self.files_done} / {self.total_files} files"
            )


            self.info_label.config(
                text=(
                    f"Finished {filename} -> {out_path}\n"
                    f"Took {file_elapsed:.1f}s"
                )
            )
            if self.files_done == self.total_files:
                self.current_file_label.config(
                    text="Current file: Done."
                )

        elif msg_type == "all_done":
            _, total_elapsed = msg
            self.info_label.config(
                text=f"All files transcribed in {total_elapsed:.1f}s."
            )
            self.is_transcribing = False

            # Re-enable controls (files remain selected, so user can re-run if they want)
            self.select_button.config(state="normal")
            self.start_button.config(state="normal")
            self.change_output_button.config(state="normal")

        elif msg_type == "error":
            _, err_text = msg
            self.is_transcribing = False
            self.select_button.config(state="normal")
            self.start_button.config(state="normal")
            self.change_output_button.config(state="normal")
            messagebox.showerror("Error", f"An error occurred:\n\n{err_text}")
            self.info_label.config(text="Error occurred. See message box.")

    def run(self):
        self.root.mainloop()


def main():
    root = tk.Tk()
    app = TranscriberApp(root)
    app.run()


if __name__ == "__main__":
    main()
