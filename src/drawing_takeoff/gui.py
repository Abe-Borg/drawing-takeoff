"""Thin desktop front-end over the headless ``extract_takeoff`` engine (M4).

Drag in (or browse to) a set of vector PDFs, confirm the scale, run, and save
the takeoff CSV. The GUI owns no measurement logic — it runs
:func:`drawing_takeoff.pipeline.extract_takeoff` on a worker thread, passes a
``progress(done, total, label)`` callback, and marshals results back to the UI
thread via ``self.after``. Mirrors the sibling project's GUI/engine seam.

The ``customtkinter`` / ``tkinterdnd2`` imports are guarded so importing this
module (and the rest of the package) never requires the ``[gui]`` extra; run
``pip install -e ".[gui]"`` then ``python -m drawing_takeoff.gui``.
"""
from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path

from . import export
from .client import get_client
from .core import api_key_store
from .pipeline import extract_takeoff

try:  # the GUI extras are optional; the engine never needs them
    import customtkinter as ctk
    from tkinter import filedialog, messagebox

    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD  # drag-and-drop is best-effort
        _DND = True
    except Exception:  # pragma: no cover - optional within the optional extra
        _DND = False
    _GUI = True
except Exception:  # pragma: no cover - exercised only without the [gui] extra
    _GUI = False


if _GUI:
    # Drag-and-drop on a customtkinter root requires inheriting tkinterdnd2's
    # DnDWrapper *and* loading the tkdnd Tcl package into the interpreter before
    # any drop target is registered (see __init__). Fall back to a plain CTk
    # root when the extra isn't present.
    _APP_BASES = (ctk.CTk, TkinterDnD.DnDWrapper) if _DND else (ctk.CTk,)

    class TakeoffApp(*_APP_BASES):
        """A small drag-drop -> takeoff-CSV window."""

        def __init__(self) -> None:
            super().__init__()
            # Load tkdnd NOW, before registering any drop target — otherwise
            # tkinterdnd2 raises 'invalid command name "tkdnd::drop_target"'.
            # Best-effort: the window still opens (browse-only) if it can't load.
            self._dnd = False
            if _DND:
                try:
                    self.TkdndVersion = TkinterDnD._require(self)
                    self._dnd = True
                except Exception:
                    self._dnd = False

            self.title("drawing-takeoff")
            self.geometry("760x640")
            self._pdfs: list[str] = []
            self._result = None
            self._events: "queue.Queue[tuple]" = queue.Queue()

            pad = {"padx": 10, "pady": 6}

            # --- API key ---------------------------------------------------
            key_row = ctk.CTkFrame(self)
            key_row.pack(fill="x", **pad)
            ctk.CTkLabel(key_row, text="Anthropic API key").pack(side="left", padx=6)
            self._key = ctk.CTkEntry(key_row, show="*", width=380)
            self._key.pack(side="left", padx=6, expand=True, fill="x")
            self._key.insert(0, api_key_store.load_api_key_from_file())
            ctk.CTkButton(key_row, text="Save", width=70, command=self._save_key).pack(side="left", padx=6)

            # --- files -----------------------------------------------------
            files = ctk.CTkFrame(self)
            files.pack(fill="both", expand=True, **pad)
            top = ctk.CTkFrame(files)
            top.pack(fill="x")
            ctk.CTkLabel(top, text="Sheets (drag PDFs here or browse)").pack(side="left", padx=6)
            ctk.CTkButton(top, text="Add PDFs…", width=100, command=self._browse).pack(side="right", padx=4)
            ctk.CTkButton(top, text="Clear", width=70, command=self._clear).pack(side="right", padx=4)
            self._filebox = ctk.CTkTextbox(files, height=120)
            self._filebox.pack(fill="both", expand=True, pady=6)
            if self._dnd:
                # Register on the root window so a PDF dropped anywhere is accepted.
                try:
                    self.drop_target_register(DND_FILES)
                    self.dnd_bind("<<Drop>>", self._on_drop)
                except Exception:
                    self._dnd = False

            # --- options ---------------------------------------------------
            opt = ctk.CTkFrame(self)
            opt.pack(fill="x", **pad)
            ctk.CTkLabel(opt, text="Scale (blank = auto-detect)").pack(side="left", padx=6)
            self._scale = ctk.CTkEntry(opt, width=130, placeholder_text="1/8\" = 1'-0\"")
            self._scale.pack(side="left", padx=6)
            ctk.CTkLabel(opt, text="Discipline").pack(side="left", padx=6)
            self._discipline = ctk.CTkEntry(opt, width=160, placeholder_text="fire protection")
            self._discipline.pack(side="left", padx=6)

            # --- run + progress -------------------------------------------
            run = ctk.CTkFrame(self)
            run.pack(fill="x", **pad)
            self._run_btn = ctk.CTkButton(run, text="Run takeoff", command=self._run)
            self._run_btn.pack(side="left", padx=6)
            self._save_btn = ctk.CTkButton(run, text="Save CSV…", command=self._save, state="disabled")
            self._save_btn.pack(side="left", padx=6)
            self._progress = ctk.CTkProgressBar(run)
            self._progress.set(0)
            self._progress.pack(side="left", padx=10, expand=True, fill="x")

            self._status = ctk.CTkLabel(self, text="Add sheets and confirm the scale.", anchor="w")
            self._status.pack(fill="x", padx=16)
            self._results = ctk.CTkTextbox(self)
            self._results.pack(fill="both", expand=True, **pad)

        # ----- file handling ---------------------------------------------
        def _add(self, paths) -> None:
            for p in paths:
                p = p.strip().strip("{}")  # tkinter dnd wraps spaced paths in braces
                if p.lower().endswith(".pdf") and p not in self._pdfs:
                    self._pdfs.append(p)
            self._filebox.delete("1.0", "end")
            self._filebox.insert("1.0", "\n".join(Path(p).name for p in self._pdfs))
            self._status.configure(text=f"{len(self._pdfs)} sheet(s) ready.")

        def _browse(self) -> None:
            self._add(filedialog.askopenfilenames(filetypes=[("PDF", "*.pdf")]))

        def _on_drop(self, event) -> None:  # pragma: no cover - GUI event
            self._add(self.tk.splitlist(event.data))

        def _clear(self) -> None:
            self._pdfs.clear()
            self._filebox.delete("1.0", "end")
            self._status.configure(text="Cleared.")

        def _save_key(self) -> None:
            try:
                api_key_store.save_api_key(self._key.get())
                messagebox.showinfo("drawing-takeoff", "API key saved.")
            except Exception as exc:
                messagebox.showerror("drawing-takeoff", f"Could not save key: {exc}")

        # ----- run on a worker thread ------------------------------------
        def _run(self) -> None:
            if not self._pdfs:
                messagebox.showwarning("drawing-takeoff", "Add at least one PDF.")
                return
            import os

            key = self._key.get().strip()
            if key:
                os.environ["ANTHROPIC_API_KEY"] = key
            self._run_btn.configure(state="disabled")
            self._save_btn.configure(state="disabled")
            self._results.delete("1.0", "end")
            self._progress.set(0)
            scale = self._scale.get().strip() or None
            disc = self._discipline.get().strip() or "construction"
            threading.Thread(target=self._worker, args=(list(self._pdfs), scale, disc), daemon=True).start()
            self.after(100, self._drain)

        def _worker(self, pdfs, scale, discipline) -> None:
            try:
                def progress(done, total, label):
                    self._events.put(("progress", done, total, label))

                result = extract_takeoff(
                    pdfs, client=get_client(), progress=progress,
                    scale_label=scale, discipline=discipline,
                )
                self._events.put(("done", result))
            except Exception:
                self._events.put(("error", traceback.format_exc()))

        def _drain(self) -> None:
            try:
                while True:
                    evt = self._events.get_nowait()
                    if evt[0] == "progress":
                        _, done, total, label = evt
                        self._progress.set(done / total if total else 0)
                        self._status.configure(text=f"[{done}/{total}] {label}")
                    elif evt[0] == "done":
                        self._finish(evt[1])
                        return
                    elif evt[0] == "error":
                        self._status.configure(text="Failed.")
                        self._results.insert("end", evt[1])
                        self._run_btn.configure(state="normal")
                        return
            except queue.Empty:
                pass
            self.after(100, self._drain)

        def _finish(self, result) -> None:
            self._result = result
            self._run_btn.configure(state="normal")
            self._save_btn.configure(state="normal")
            self._progress.set(1)
            lines = [f"=== Takeoff: {result.sheet_count} sheet(s) ==="]
            if result.per_system_totals:
                lines.append("\nTOTALS by system:")
                lines += [f"  {s}: {q:,.1f} LF" for s, q in result.per_system_totals.items()]
            else:
                lines.append("\n(no confidently-measured systems)")
            if result.flagged:
                lines.append("\nFLAGGED for review (not counted):")
                lines += [
                    f"  {it.sheet}  {it.quantity:,.1f} LF -> {it.system} "
                    f"[{it.confidence}{', ambiguous' if it.ambiguous else ''}]"
                    for it in result.flagged
                ]
            if result.errors:
                lines.append("\nERRORS:")
                lines += [f"  {e}" for e in result.errors]
            self._results.insert("1.0", "\n".join(lines))
            self._status.configure(text="Done. Save the CSV to keep it.")

        def _save(self) -> None:
            if self._result is None:
                return
            out = filedialog.askdirectory(title="Choose a folder for the takeoff export")
            if not out:
                return
            folder = export.write_takeoff_export(self._result, out, project_name="takeoff")
            messagebox.showinfo("drawing-takeoff", f"Saved to:\n{folder}")

    def main() -> int:
        ctk.set_appearance_mode("system")
        TakeoffApp().mainloop()
        return 0

else:  # pragma: no cover - only when the [gui] extra is not installed

    def main() -> int:
        raise SystemExit(
            "The GUI needs the optional extras. Install them with:\n"
            '    pip install -e ".[gui]"\n'
            "then run:  python -m drawing_takeoff.gui"
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
