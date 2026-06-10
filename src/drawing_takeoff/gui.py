"""Thin desktop front-end over the headless takeoff engines (M4 + M7).

Drag in (or browse to) a set of vector PDFs, pick an output mode, confirm the
scale, run, and save. The GUI owns no measurement logic — it runs one of the two
engine entry points on a worker thread, passes a ``progress(done, total, label)``
callback, and marshals results back to the UI thread via ``self.after``:

  * **By system** — :func:`drawing_takeoff.pipeline.extract_takeoff`: trusted
    linear feet totaled per system across the set, saved as CSVs.
  * **By system × size** — :func:`drawing_takeoff.pipeline.extract_system_size_takeoff`:
    pipe-network detection + size callouts + a high-DPI second-look re-check,
    totaled by system AND nominal size, saved as an Excel workbook + one
    marked-up PDF per sheet (the ``legend --system-size`` CLI path).

An optional advisory legend (a lead-sheet PDF/image) attaches to either mode.

The ``customtkinter`` / ``tkinterdnd2`` imports are guarded so importing this
module (and the rest of the package) never requires the ``[gui]`` extra; run
``pip install -e ".[gui]"`` then ``python -m drawing_takeoff.gui``.
"""
from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path

from . import export, legend
from .client import get_client
from .core import api_key_store
from .models import SystemSizeResult
from .pipeline import extract_system_size_takeoff, extract_takeoff, write_system_size_export

try:  # the GUI extras are optional; the engine never needs them
    import customtkinter as ctk
    from tkinter import BooleanVar, filedialog, messagebox

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
        """A small drag-drop -> takeoff window with two output modes."""

        # Segmented-button label -> internal mode key.
        _MODES = {"By system": "system", "By system × size": "size"}

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
            self.geometry("820x720")
            self._pdfs: list[str] = []
            self._mode = "system"
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

            # --- output mode ----------------------------------------------
            mode_row = ctk.CTkFrame(self)
            mode_row.pack(fill="x", **pad)
            ctk.CTkLabel(mode_row, text="Output").pack(side="left", padx=6)
            self._mode_btn = ctk.CTkSegmentedButton(
                mode_row, values=list(self._MODES), command=self._on_mode
            )
            self._mode_btn.set("By system")
            self._mode_btn.pack(side="left", padx=6)
            ctk.CTkLabel(
                mode_row,
                text="system × size adds pipe-network detection, sizes, Excel + marked-up PDF",
                text_color="gray",
            ).pack(side="left", padx=10)

            # --- files -----------------------------------------------------
            files = ctk.CTkFrame(self)
            files.pack(fill="both", expand=True, **pad)
            top = ctk.CTkFrame(files)
            top.pack(fill="x")
            ctk.CTkLabel(top, text="Sheets (drag PDFs here or browse)").pack(side="left", padx=6)
            ctk.CTkButton(top, text="Add PDFs…", width=100, command=self._browse).pack(side="right", padx=4)
            ctk.CTkButton(top, text="Clear", width=70, command=self._clear).pack(side="right", padx=4)
            self._filebox = ctk.CTkTextbox(files, height=110)
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

            # --- advisory legend (both modes) ------------------------------
            leg = ctk.CTkFrame(self)
            leg.pack(fill="x", **pad)
            ctk.CTkLabel(leg, text="Legend (advisory)").pack(side="left", padx=6)
            self._legend = ctk.CTkEntry(leg, placeholder_text="optional lead-sheet PDF/image with the legend/symbols")
            self._legend.pack(side="left", padx=6, expand=True, fill="x")
            ctk.CTkLabel(leg, text="page").pack(side="left", padx=(6, 2))
            self._legend_page = ctk.CTkEntry(leg, width=46)
            self._legend_page.insert(0, "0")
            self._legend_page.pack(side="left", padx=(0, 6))
            ctk.CTkButton(leg, text="Choose…", width=80, command=self._browse_legend).pack(side="left", padx=4)
            ctk.CTkButton(leg, text="Clear", width=60,
                          command=lambda: self._legend.delete(0, "end")).pack(side="left", padx=4)

            # --- system × size options (shown only in that mode) ----------
            self._size_frame = ctk.CTkFrame(self)
            ctk.CTkLabel(self._size_frame, text="System × Size:").pack(side="left", padx=6)
            ctk.CTkLabel(self._size_frame, text="top networks").pack(side="left", padx=(6, 2))
            self._top = ctk.CTkEntry(self._size_frame, width=56)
            self._top.insert(0, "8")
            self._top.pack(side="left", padx=(0, 8))
            ctk.CTkLabel(self._size_frame, text="max styles").pack(side="left", padx=(6, 2))
            self._max_styles = ctk.CTkEntry(self._size_frame, width=56)
            self._max_styles.insert(0, "12")
            self._max_styles.pack(side="left", padx=(0, 8))
            self._second_look = BooleanVar(value=True)
            ctk.CTkCheckBox(self._size_frame, text="Second look (re-check flagged)",
                            variable=self._second_look).pack(side="left", padx=8)

            # --- run + progress -------------------------------------------
            self._run_frame = ctk.CTkFrame(self)
            self._run_frame.pack(fill="x", **pad)
            self._run_btn = ctk.CTkButton(self._run_frame, text="Run takeoff", command=self._run)
            self._run_btn.pack(side="left", padx=6)
            self._save_btn = ctk.CTkButton(self._run_frame, text="Save CSV…", command=self._save, state="disabled")
            self._save_btn.pack(side="left", padx=6)
            self._progress = ctk.CTkProgressBar(self._run_frame)
            self._progress.set(0)
            self._progress.pack(side="left", padx=10, expand=True, fill="x")

            self._status = ctk.CTkLabel(self, text="Add sheets and confirm the scale.", anchor="w")
            self._status.pack(fill="x", padx=16)
            self._results = ctk.CTkTextbox(self)
            self._results.pack(fill="both", expand=True, **pad)

        # ----- mode ------------------------------------------------------
        def _on_mode(self, value: str) -> None:
            self._mode = self._MODES.get(value, "system")
            if self._mode == "size":
                self._size_frame.pack(fill="x", padx=10, pady=6, before=self._run_frame)
                self._save_btn.configure(text="Save Excel + PDF…")
                self._status.configure(
                    text="System × Size: detects pipe networks per sheet, totals LF by system and nominal size."
                )
            else:
                self._size_frame.pack_forget()
                self._save_btn.configure(text="Save CSV…")
                self._status.configure(
                    text="By system: totals trusted linear feet per system across the set."
                )

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

        def _browse_legend(self) -> None:
            path = filedialog.askopenfilename(
                title="Choose the lead-sheet legend (advisory)",
                filetypes=[
                    ("PDF or image", "*.pdf *.png *.jpg *.jpeg *.webp *.gif *.bmp *.tif *.tiff"),
                    ("All files", "*.*"),
                ],
            )
            if path:
                self._legend.delete(0, "end")
                self._legend.insert(0, path)

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

        @staticmethod
        def _int(entry, default: int) -> int:
            try:
                return int(entry.get().strip())
            except (TypeError, ValueError):
                return default

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
            legend_path = self._legend.get().strip() or None
            opts = {
                "legend_path": legend_path,
                "legend_page": self._int(self._legend_page, 0),
                "top": self._int(self._top, 8),
                "max_styles": self._int(self._max_styles, 12),
                "second_look": bool(self._second_look.get()),
            }
            threading.Thread(
                target=self._worker, args=(list(self._pdfs), scale, disc, self._mode, opts), daemon=True
            ).start()
            self.after(100, self._drain)

        def _worker(self, pdfs, scale, discipline, mode, opts) -> None:
            try:
                def progress(done, total, label):
                    self._events.put(("progress", done, total, label))

                legend_pdf = legend_image = None
                if opts["legend_path"]:
                    kind, data = legend._load_legend_attachment(opts["legend_path"], opts["legend_page"])
                    legend_pdf = data if kind == "pdf" else None
                    legend_image = data if kind == "image" else None

                client = get_client()
                if mode == "size":
                    result = extract_system_size_takeoff(
                        pdfs, client=client, progress=progress, scale_label=scale, discipline=discipline,
                        legend_pdf=legend_pdf, legend_image=legend_image,
                        top=opts["top"], max_styles=opts["max_styles"], second_look=opts["second_look"],
                    )
                else:
                    result = extract_takeoff(
                        pdfs, client=client, progress=progress, scale_label=scale, discipline=discipline,
                        legend_pdf=legend_pdf, legend_image=legend_image,
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
            text = (
                self._render_system_size(result)
                if isinstance(result, SystemSizeResult)
                else self._render_by_system(result)
            )
            self._results.insert("1.0", text)
            self._status.configure(text="Done. Save to keep the output.")

        def _render_by_system(self, result) -> str:
            lines = [f"=== Takeoff by system: {result.sheet_count} sheet(s) ==="]
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
            return "\n".join(lines)

        def _render_system_size(self, result) -> str:
            lines = [f"=== Takeoff by system × size: {result.sheet_count} sheet(s) ==="]
            totals = result.per_system_totals
            if totals:
                lines.append("\nTOTALS by system:")
                lines += [f"  {s}: {q:,.1f} LF" for s, q in totals.items()]
                systems: dict[str, dict[str, float]] = {}
                for (system, size), lf in result.by_system_size.items():
                    systems.setdefault(system, {})[size] = lf
                lines.append("\nBY SYSTEM × SIZE (linear feet):")
                for system in sorted(systems, key=lambda s: -sum(systems[s].values())):
                    lines.append(f"  {system}: {sum(systems[system].values()):,.1f} LF")
                    for size, lf in sorted(systems[system].items(),
                                           key=lambda kv: export._SIZE_ORDER.get(kv[0], 1e9)):
                        lines.append(f"      {size:>10}  {lf:>9,.1f} LF")
            else:
                lines.append("\n(no trusted pipe networks)")
            if result.review:
                lines.append("\nREVIEW (not counted / confirm — see the Review tab):")
                lines += [f"  {r}" for r in result.review]
            if result.errors:
                lines.append("\nERRORS:")
                lines += [f"  {e}" for e in result.errors]
            return "\n".join(lines)

        def _save(self) -> None:
            if self._result is None:
                return
            if isinstance(self._result, SystemSizeResult):
                out = filedialog.askdirectory(title="Choose a folder for the System × Size export")
                if not out:
                    return
                folder = write_system_size_export(self._result, out, project_name="takeoff")
                messagebox.showinfo("drawing-takeoff", f"Saved Excel + marked-up PDF(s) to:\n{folder}")
            else:
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
