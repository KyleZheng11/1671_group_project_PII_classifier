"""
PII Detector GUI
Wraps the token-classification pipeline in a Tkinter window with:
  - text input, file loading, and keyboard shortcut (Ctrl+Enter) to analyze
  - color-coded highlighting of detected PII spans + a legend with counts
  - hover tooltips showing label/confidence for each highlight
  - a results table (click a row to jump to that span)
  - a confidence threshold slider
  - optional appending of results to the same CSV as before
  - redaction of chosen PII categories with a live preview and .txt export
"""

import csv
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

CSV_PATH = "output/pii_results.csv"
FIELDS = ["start", "end", "text", "label", "score"]
MODEL_PATH = "./bert_pii_model_full"

# Colors are assigned to labels in the order they first appear
PALETTE = [
    "#FFB3BA", "#BAE1FF", "#BAFFC9", "#FFFFBA", "#FFDFBA",
    "#E0BBFF", "#C9F0FF", "#FFC8E8", "#D4F4C4", "#F0E0C0",
]
HOVER_HINT = "Hover over a highlight to see its label and confidence."

# Redaction (same categories and marker as redact.py)
CATEGORIES = [
    "first_name",
    "last_name",
    "email",
    "street_address",
    "phone_number",
    "date_of_birth",
    "company_name",
    "other",
]
REDACTION = "[REDACTED]"


class PIIApp:
    def __init__(self, root):
        self.root = root
        root.title("PII Detector")
        root.geometry("1100x760")
        root.minsize(800, 600)

        self.pii = None
        self.q = queue.Queue()          # worker threads -> UI thread
        self.label_colors = {}
        self.last_text = ""
        self.last_results = []          # every result from the last run
        self.shown = []                 # results above the threshold
        self.redacted_text = ""
        self.redacted_count = 0
        self.loaded_path = None         # used to suggest an export filename
        self.cat_vars = {}              # label -> BooleanVar for redaction

        self._build_ui()
        self._update_redaction()
        self._set_busy(True, "Loading model… (this can take a moment)")
        threading.Thread(target=self._load_model, daemon=True).start()
        self.root.after(100, self._poll)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # Toolbar
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", **pad)
        self.analyze_btn = ttk.Button(bar, text="Analyze  (Ctrl+Enter)", command=self.analyze)
        self.analyze_btn.pack(side="left")
        ttk.Button(bar, text="Load .txt file…", command=self.load_file).pack(side="left", padx=6)
        ttk.Button(bar, text="Clear", command=self.clear).pack(side="left")
        self.save_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text=f"Append results to {CSV_PATH}",
                        variable=self.save_var).pack(side="right")

        # Input
        ttk.Label(self.root, text="Input text", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=8)
        in_frame = ttk.Frame(self.root)
        in_frame.pack(fill="x", padx=8)
        self.input = tk.Text(in_frame, height=8, wrap="word", undo=True, font=("Consolas", 11))
        in_scroll = ttk.Scrollbar(in_frame, command=self.input.yview)
        self.input.configure(yscrollcommand=in_scroll.set)
        self.input.pack(side="left", fill="both", expand=True)
        in_scroll.pack(side="right", fill="y")
        self.input.bind("<Control-Return>", lambda e: (self.analyze(), "break")[1])

        # Threshold slider
        ctl = ttk.Frame(self.root)
        ctl.pack(fill="x", **pad)
        ttk.Label(ctl, text="Minimum confidence:").pack(side="left")
        self.threshold = tk.DoubleVar(value=0.0)
        ttk.Scale(ctl, from_=0.0, to=1.0, variable=self.threshold, length=220,
                  command=lambda _v: self._on_threshold()).pack(side="left", padx=6)
        self.thr_label = ttk.Label(ctl, text="0.00", width=5)
        self.thr_label.pack(side="left")

        # Legend
        ttk.Label(self.root, text="Legend", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=8)
        self.legend = ttk.Frame(self.root)
        self.legend.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Label(self.legend, text="Run an analysis to see detected PII types.",
                  foreground="gray").pack(side="left")

        # Redaction controls
        red = ttk.LabelFrame(self.root, text="Redact these PII types")
        red.pack(fill="x", padx=8, pady=4)
        self.cat_frame = ttk.Frame(red)
        self.cat_frame.pack(fill="x", padx=4, pady=2)
        for cat in CATEGORIES:
            self._add_category(cat)
        red_btns = ttk.Frame(red)
        red_btns.pack(fill="x", padx=4, pady=(2, 4))
        ttk.Button(red_btns, text="Select all", command=lambda: self._set_all_cats(True)).pack(side="left")
        ttk.Button(red_btns, text="Select none", command=lambda: self._set_all_cats(False)).pack(side="left", padx=6)
        self.redact_btn = ttk.Button(red_btns, text="Redact", command=self.show_redacted)
        self.redact_btn.pack(side="left", padx=(12, 0))
        self.export_btn = ttk.Button(red_btns, text="Export redacted file…", command=self.export_redacted)
        self.export_btn.pack(side="left", padx=6)
        self.redact_info = ttk.Label(red_btns, text="", foreground="#444")
        self.redact_info.pack(side="left", padx=8)

        # Results: highlighted / redacted text (left) | table (right)
        panes = ttk.PanedWindow(self.root, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(panes)
        self.views = ttk.Notebook(left)
        self.views.pack(fill="both", expand=True)

        self.out = tk.Text(self.views, wrap="word", state="disabled", font=("Consolas", 11),
                           cursor="arrow", background="#FAFAFA")
        self.views.add(self.out, text="Highlighted text")

        self.red_out = tk.Text(self.views, wrap="word", state="disabled", font=("Consolas", 11),
                               background="#FAFAFA")
        self.red_out.tag_configure("redacted", background="#333333", foreground="white")
        self.views.add(self.red_out, text="Redacted text")
        self.out.tag_configure("focus", underline=True, foreground="#B00020",
                               font=("Consolas", 11, "bold"))
        self.out.bind("<Motion>", self._on_hover)
        self.out.bind("<Leave>", lambda e: self.hover_var.set(HOVER_HINT))
        self.hover_var = tk.StringVar(value=HOVER_HINT)
        ttk.Label(left, textvariable=self.hover_var, foreground="#444").pack(anchor="w", pady=(2, 0))
        panes.add(left, weight=3)

        right = ttk.Frame(panes)
        ttk.Label(right, text="Detections (click to locate)",
                  font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        cols = ("start", "end", "text", "label", "score")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", selectmode="browse")
        widths = {"start": 55, "end": 55, "text": 180, "label": 110, "score": 60}
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=widths[c], anchor="w" if c in ("text", "label") else "e")
        tree_scroll = ttk.Scrollbar(right, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        panes.add(right, weight=2)

        # Status bar
        self.status = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status, relief="sunken",
                  anchor="w").pack(fill="x", side="bottom")

    # ------------------------------------------------------------- workers
    def _load_model(self):
        try:
            from transformers import pipeline
            p = pipeline(
                "token-classification",
                model=MODEL_PATH,
                aggregation_strategy="first",
                stride=128,
            )
            self.q.put(("model_ready", p))
        except Exception as e:
            self.q.put(("fatal", f"Could not load model from {MODEL_PATH}:\n{e}"))

    def _run_model(self, text):
        try:
            self.q.put(("results", (text, self.pii(text))))
        except Exception as e:
            self.q.put(("error", f"Analysis failed:\n{e}"))

    def _poll(self):
        """Handle messages from worker threads on the UI thread."""
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "model_ready":
                    self.pii = payload
                    self._set_busy(False, "Model loaded. Enter text and press Analyze.")
                elif kind == "results":
                    self._handle_results(*payload)
                elif kind == "error":
                    self._set_busy(False, "Error — see message.")
                    messagebox.showerror("PII Detector", payload)
                elif kind == "fatal":
                    self._set_busy(True, "Model failed to load.")
                    messagebox.showerror("PII Detector", payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # ------------------------------------------------------------- actions
    def analyze(self):
        if self.pii is None:
            return
        text = self.input.get("1.0", "end-1c")
        if not text.strip():
            self.status.set("Nothing to analyze — enter some text first.")
            return
        self._set_busy(True, "Analyzing…")
        threading.Thread(target=self._run_model, args=(text,), daemon=True).start()

    def load_file(self):
        path = filedialog.askopenfilename(
            title="Open text file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as e:
            messagebox.showerror("PII Detector", f"Could not read file:\n{e}")
            return
        self.input.delete("1.0", "end")
        self.input.insert("1.0", content)
        self.loaded_path = path
        self.status.set(f"Loaded {os.path.basename(path)} ({len(content):,} chars). Press Analyze.")

    def clear(self):
        self.input.delete("1.0", "end")
        self.last_text, self.last_results = "", []
        self.loaded_path = None
        self._render()
        self.status.set("Cleared.")

    # ------------------------------------------------------------- results
    def _handle_results(self, text, results):
        self.last_text = text
        self.last_results = [{
            "start": r["start"],
            "end": r["end"],
            "text": text[r["start"]:r["end"]],
            "label": r["entity_group"],
            "score": float(r["score"]),
        } for r in results]
        self._render()

        msg = f"Found {len(self.last_results)} entit{'y' if len(self.last_results) == 1 else 'ies'}"
        msg += f" ({len(self.shown)} above threshold)."
        if self.save_var.get() and self.shown:
            try:
                self._save(self.shown)
                msg += f" Saved {len(self.shown)} row(s) to {CSV_PATH}."
            except OSError as e:
                messagebox.showerror("PII Detector", f"Could not write CSV:\n{e}")
        self._set_busy(False, msg)

    def _save(self, rows):
        os.makedirs(os.path.dirname(CSV_PATH) or ".", exist_ok=True)
        write_header = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                writer.writeheader()
            for r in rows:
                writer.writerow({**r, "score": round(r["score"], 4)})

    def _color_for(self, label):
        if label not in self.label_colors:
            color = PALETTE[len(self.label_colors) % len(PALETTE)]
            self.label_colors[label] = color
            self.out.tag_configure(f"lbl_{label}", background=color)
            self.tree.tag_configure(label, background=color)
            self.out.tag_raise("focus")
        return self.label_colors[label]

    def _render(self):
        thr = self.threshold.get()
        self.shown = [r for r in self.last_results if r["score"] >= thr]

        # Highlighted text
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", self.last_text)
        for r in self.shown:
            self._color_for(r["label"])
            self.out.tag_add(f"lbl_{r['label']}", f"1.0+{r['start']}c", f"1.0+{r['end']}c")
        self.out.configure(state="disabled")

        # Table
        self.tree.delete(*self.tree.get_children())
        for i, r in enumerate(self.shown):
            self.tree.insert("", "end", iid=str(i), tags=(r["label"],),
                             values=(r["start"], r["end"], r["text"], r["label"], f"{r['score']:.2f}"))

        # Legend with counts
        for w in self.legend.winfo_children():
            w.destroy()
        counts = {}
        for r in self.shown:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        if not self.last_text:
            ttk.Label(self.legend, text="Run an analysis to see detected PII types.",
                      foreground="gray").pack(side="left")
        elif not counts:
            ttk.Label(self.legend, text="No PII found.", foreground="green").pack(side="left")
        else:
            for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                tk.Label(self.legend, text=f" {label}  ×{n} ", bg=self._color_for(label),
                         relief="solid", borderwidth=1, padx=4).pack(side="left", padx=3)

        # Labels the model returns that aren't in CATEGORIES still get a checkbox
        for r in self.last_results:
            if r["label"] not in self.cat_vars:
                self._add_category(r["label"])

        self._update_redaction()

    # ----------------------------------------------------------- redaction
    def _add_category(self, label):
        var = tk.BooleanVar(value=True)
        self.cat_vars[label] = var
        ttk.Checkbutton(self.cat_frame, text=label.replace("_", " "), variable=var,
                        command=self._update_redaction).pack(side="left", padx=(0, 10))

    def _set_all_cats(self, value):
        for var in self.cat_vars.values():
            var.set(value)
        self._update_redaction()

    def _redact(self, text, results, labels):
        """Replace every span whose label is in `labels` with REDACTION (as in redact.py).
        Returns (redacted_text, number_of_spans, list of (start, end) in the new text)."""
        spans = sorted((r["start"], r["end"]) for r in results if r["label"] in labels)

        # Merge overlapping spans so nothing gets replaced twice
        merged = []
        for start, end in spans:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

        # Build the output left to right, tracking where each marker lands
        parts, marks, pos, out_len = [], [], 0, 0
        for start, end in merged:
            parts.append(text[pos:start])
            out_len += start - pos
            marks.append((out_len, out_len + len(REDACTION)))
            parts.append(REDACTION)
            out_len += len(REDACTION)
            pos = end
        parts.append(text[pos:])
        return "".join(parts), len(merged), marks

    def _update_redaction(self):
        """Recompute the redacted view from the current results, threshold and checkboxes."""
        labels = {lbl for lbl, var in self.cat_vars.items() if var.get()}
        self.redacted_text, self.redacted_count, marks = self._redact(
            self.last_text, self.shown, labels)

        self.red_out.configure(state="normal")
        self.red_out.delete("1.0", "end")
        self.red_out.insert("1.0", self.redacted_text)
        for s, e in marks:
            self.red_out.tag_add("redacted", f"1.0+{s}c", f"1.0+{e}c")
        self.red_out.configure(state="disabled")

        has_text = bool(self.last_text)
        self.export_btn.state(["!disabled"] if has_text else ["disabled"])
        self.redact_btn.state(["!disabled"] if has_text else ["disabled"])
        self.redact_info.configure(
            text=f"{self.redacted_count} span(s) redacted" if has_text else "Analyze text first.")

    def show_redacted(self):
        self._update_redaction()
        self.views.select(self.red_out)
        self.status.set(f"Redacted {self.redacted_count} span(s). Use Export to save the file.")

    def export_redacted(self):
        if not self.last_text:
            return
        if self.loaded_path:
            base, _ext = os.path.splitext(os.path.basename(self.loaded_path))
            initial_dir = os.path.dirname(self.loaded_path)
        else:
            base, initial_dir = "text", os.getcwd()
        path = filedialog.asksaveasfilename(
            title="Export redacted file",
            initialdir=initial_dir,
            initialfile=f"{base}_redacted.txt",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.redacted_text)
        except OSError as e:
            messagebox.showerror("PII Detector", f"Could not save file:\n{e}")
            return
        self.status.set(f"Exported redacted text ({self.redacted_count} span(s)) to {path}")

    def _on_threshold(self):
        self.thr_label.configure(text=f"{self.threshold.get():.2f}")
        if self.last_text:
            self._render()
            self.status.set(f"Showing {len(self.shown)} of {len(self.last_results)} "
                            f"entities at confidence ≥ {self.threshold.get():.2f}.")

    def _on_select(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        r = self.shown[int(sel[0])]
        start, end = f"1.0+{r['start']}c", f"1.0+{r['end']}c"
        self.out.tag_remove("focus", "1.0", "end")
        self.out.tag_add("focus", start, end)
        self.out.see(start)

    def _on_hover(self, event):
        idx = self.out.index(f"@{event.x},{event.y}")
        count = self.out.count("1.0", idx, "chars")
        offset = count[0] if count else 0
        for r in self.shown:
            if r["start"] <= offset < r["end"]:
                self.hover_var.set(f'{r["label"]}  ·  confidence {r["score"]:.2f}  ·  "{r["text"]}"')
                return
        self.hover_var.set(HOVER_HINT)

    # -------------------------------------------------------------- helpers
    def _set_busy(self, busy, msg):
        self.analyze_btn.state(["disabled"] if busy else ["!disabled"])
        self.root.configure(cursor="watch" if busy else "")
        self.status.set(msg)


if __name__ == "__main__":
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    PIIApp(root)
    root.mainloop()