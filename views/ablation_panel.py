"""
Ablation Panel — multi-seed feature ablation experiment UI.

Lets the user run ``ModelManager.run_multi_seed_ablation()`` from the GUI,
displays summary results with per-metric verdict badges, and persists the
last result to ``ablation_result.json`` for reload on next open.
"""
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import List, Optional, Dict, Any

import pandas as pd

from models.model_manager import ModelManager


class AblationPanel(tk.Frame):
    """Tab panel for running and viewing multi-seed ablation experiments."""

    # ── Tuneable constants ────────────────────────────────────────
    VERDICT_THRESHOLD  = 0.5   # pooled-std effect size to declare a winner
    SEED_COUNT_DEFAULT = 10
    SEED_COUNT_MIN     = 2
    SEED_COUNT_MAX     = 100
    SEED_COUNT_WARN    = 50    # show timing warning above this

    _BADGE = {
        'KEEP':          ('✓ KEEP', 'good'),
        'REMOVE':        ('✗ REMOVE', 'bad'),
        'INCONCLUSIVE':  ('~ INCONCLUSIVE', 'warn'),
    }

    def __init__(self, parent, model_manager: ModelManager, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.mm = model_manager
        self._featured_df: Optional[pd.DataFrame] = None
        self._featured_df_id: Optional[int] = None   # id() of last pushed df
        self._running = False
        self._build_ui()

        # Try to load previous result on open
        prev = self.mm.load_ablation_result()
        if prev:
            self._render_result(prev)

    # ── Public API called by MainWindow ──────────────────────────

    def set_featured_df(self, df: pd.DataFrame) -> None:
        """Supply the feature-engineered DataFrame.

        Enables the Run button once called with a non-empty frame.
        Populates the column picker listbox only when the DataFrame changes.
        """
        df_id = id(df) if df is not None else None
        if df_id == self._featured_df_id:
            return  # same object — nothing to do
        self._featured_df = df
        self._featured_df_id = df_id
        has_data = df is not None and not df.empty
        self.run_btn.config(state=tk.NORMAL if has_data else tk.DISABLED)
        if has_data:
            features = self.mm.get_feature_columns(df)
            self._populate_columns(features)
            self.status_label.config(
                text=f"Ready ({len(df)} rows, {len(features)} features)"
            )

    # ── UI construction ──────────────────────────────────────────

    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ── Top bar: run button + seed control ───────────────────
        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, sticky='ew', pady=(0, 5))

        self.run_btn = ttk.Button(
            bar, text="▶ Run Ablation", command=self._on_run, state=tk.DISABLED,
        )
        self.run_btn.pack(side=tk.LEFT, padx=5)

        ttk.Label(bar, text="Seeds:").pack(side=tk.LEFT, padx=(15, 2))
        self.seed_var = tk.IntVar(value=self.SEED_COUNT_DEFAULT)
        self.seed_spin = ttk.Spinbox(
            bar, from_=self.SEED_COUNT_MIN, to=self.SEED_COUNT_MAX,
            textvariable=self.seed_var, width=5,
        )
        self.seed_spin.pack(side=tk.LEFT)

        self.seed_warn = ttk.Label(bar, text="", foreground='orange')
        self.seed_warn.pack(side=tk.LEFT, padx=5)

        # Bind validation for seed count warning
        self.seed_var.trace_add('write', self._check_seed_count)

        self.status_label = ttk.Label(bar, text="No data loaded", foreground='grey')
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # ── Main content: columns listbox (left) + results (right) ──
        pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pane.grid(row=1, column=0, sticky='nsew')

        # Left side: feature column picker
        left = ttk.Frame(pane)
        pane.add(left, weight=1)

        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        lbl_frame = ttk.Frame(left)
        lbl_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=(2, 0))
        ttk.Label(lbl_frame, text="Features to ablate:").pack(side=tk.LEFT)
        self.sel_count_label = ttk.Label(lbl_frame, text="(0 selected)",
                                         foreground='grey')
        self.sel_count_label.pack(side=tk.RIGHT)

        list_frame = ttk.Frame(left)
        list_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=2)
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        self.col_listbox = tk.Listbox(
            list_frame, selectmode=tk.EXTENDED, exportselection=False,
        )
        self.col_listbox.grid(row=0, column=0, sticky='nsew')
        col_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                    command=self.col_listbox.yview)
        col_scroll.grid(row=0, column=1, sticky='ns')
        self.col_listbox.config(yscrollcommand=col_scroll.set)
        self.col_listbox.bind('<<ListboxSelect>>', self._on_selection_change)

        # Progress bar below the listbox
        self.progress = ttk.Progressbar(left, mode='determinate')
        self.progress.grid(row=2, column=0, sticky='ew', padx=5, pady=(2, 5))

        # Right side: results area
        right = ttk.Frame(pane)
        pane.add(right, weight=3)

        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self.text = scrolledtext.ScrolledText(
            right, wrap=tk.WORD, state=tk.DISABLED,
        )
        self.text.grid(row=0, column=0, sticky='nsew')

        # Colour tags
        self.text.tag_config('header', foreground='#1a1a2e',
                             font=('TkDefaultFont', 0, 'bold'))
        self.text.tag_config('good', foreground='green')
        self.text.tag_config('bad', foreground='red')
        self.text.tag_config('warn', foreground='orange')
        self.text.tag_config('neutral', foreground='black')
        self.text.tag_config('dim', foreground='grey')

    # ── Seed-count warning ───────────────────────────────────────

    def _check_seed_count(self, *_args):
        try:
            n = self.seed_var.get()
        except (tk.TclError, ValueError):
            return
        if n > self.SEED_COUNT_WARN:
            self.seed_warn.config(text=f"⚠ {n} seeds — this may take a while")
        else:
            self.seed_warn.config(text="")

    # ── Column picker helpers ────────────────────────────────────

    def _populate_columns(self, features: List[str]) -> None:
        """Fill the listbox with available feature column names."""
        self.col_listbox.delete(0, tk.END)
        for name in sorted(features):
            self.col_listbox.insert(tk.END, name)
        self._on_selection_change()

    def _get_selected_columns(self) -> List[str]:
        """Return the feature names currently selected in the listbox."""
        indices = self.col_listbox.curselection()
        return [self.col_listbox.get(i) for i in indices]

    def _on_selection_change(self, _event=None):
        """Update the selection count label."""
        n = len(self.col_listbox.curselection())
        self.sel_count_label.config(text=f"({n} selected)")

    # ── Run logic ────────────────────────────────────────────────

    def _on_run(self):
        if self._running or self._featured_df is None:
            return

        try:
            n_seeds = self.seed_var.get()
        except (tk.TclError, ValueError):
            n_seeds = 10

        ablate_cols = self._get_selected_columns()
        if not ablate_cols:
            self.status_label.config(text="Select at least one feature to ablate")
            return

        self._running = True
        self.run_btn.config(state=tk.DISABLED)
        self.progress['value'] = 0
        self.progress['maximum'] = n_seeds
        self.status_label.config(text="Running…")

        def worker():
            def on_progress(current, total):
                self.after(0, lambda c=current: self._update_progress(c))

            try:
                result = self.mm.run_multi_seed_ablation(
                    self._featured_df,
                    seeds=range(n_seeds),
                    ablate_cols=ablate_cols,
                    progress_cb=on_progress,
                    verdict_threshold=self.VERDICT_THRESHOLD,
                )
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.after(0, lambda: self._on_error(str(exc)))
                return
            self.after(0, lambda: self._on_complete(result))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _update_progress(self, current: int):
        self.progress['value'] = current

    def _on_error(self, message: str):
        """Handle worker-thread exception — runs on main thread via after()."""
        self._running = False
        has_data = self._featured_df is not None and not self._featured_df.empty
        self.run_btn.config(state=tk.NORMAL if has_data else tk.DISABLED)
        self.progress['value'] = 0
        self.status_label.config(text=f"Error: {message}")

    def _on_complete(self, result: Dict[str, Any]):
        self._running = False
        has_data = self._featured_df is not None and not self._featured_df.empty
        self.run_btn.config(state=tk.NORMAL if has_data else tk.DISABLED)
        self.progress['value'] = self.progress['maximum']

        if not result:
            self.status_label.config(text="Ablation failed — check log")
            return

        self.status_label.config(text=f"Done — {result.get('ran_at', '')}")
        self._render_result(result)

    # ── Rendering ────────────────────────────────────────────────

    def _render_result(self, result: Dict[str, Any]):
        self.text.config(state=tk.NORMAL)
        self.text.delete('1.0', tk.END)

        full = result.get('full_features', {})
        model = result.get('model_features', {})
        verdicts = result.get('verdicts', {})
        overall = result.get('overall', 'INCONCLUSIVE')
        ablated = result.get('ablated_cols', [])
        seeds = result.get('seeds', [])
        ran_at = result.get('ran_at', '')

        # ── Header ───────────────────────────────────────────────
        self._section("MULTI-SEED ABLATION RESULTS")
        self._put(f"  Ran: {ran_at}\n", 'dim')
        self._put(f"  Seeds: {len(seeds)}   Ablated: {', '.join(ablated)}\n\n", 'dim')

        # ── Overall verdict ──────────────────────────────────────
        badge_text, badge_tag = self._BADGE.get(overall, ('?', 'neutral'))
        self._put(f"  Overall recommendation:  ", 'neutral')
        self._put(f"{badge_text}\n\n", badge_tag)

        # ── Summary table ────────────────────────────────────────
        self._section("METRIC COMPARISON")
        self._put(
            f"  {'Metric':<10s} {'Full features':>16s} "
            f"{'Model features':>16s} {'Delta':>10s}  {'Verdict':>15s}\n",
            'header',
        )
        self._put("  " + "─" * 72 + "\n", 'dim')

        for metric in ('accuracy', 'ece', 'mce'):
            fm = full.get(f'{metric}_mean', 0)
            fs = full.get(f'{metric}_std', 0)
            mm_ = model.get(f'{metric}_mean', 0)
            ms = model.get(f'{metric}_std', 0)
            delta = fm - mm_

            v = verdicts.get(metric, 'INCONCLUSIVE')
            v_text, v_tag = self._BADGE.get(v, ('?', 'neutral'))

            self._put(
                f"  {metric:<10s} {fm:.4f}±{fs:.4f}   "
                f"{mm_:.4f}±{ms:.4f}   {delta:+.4f}  ",
                'neutral',
            )
            self._put(f"{v_text}\n", v_tag)

        self._put("\n", 'neutral')

        # ── Interpretation guide ────────────────────────────────
        self._section("INTERPRETATION")
        self._put("  Full features   = current model features + ablated columns\n", 'dim')
        self._put("  Model features  = current model features (without ablated)\n", 'dim')
        self._put("  Accuracy: higher is better   ECE/MCE: lower is better\n", 'dim')
        self._put("  KEEP = full features win on ≥2 metrics by >1 std dev\n", 'dim')
        self._put("  REMOVE = model features win on ≥2 metrics by >1 std dev\n", 'dim')
        self._put("  INCONCLUSIVE = deltas within noise or metrics split\n\n", 'dim')

        # ── Per-seed detail ──────────────────────────────────────
        per_seed = result.get('per_seed', [])
        if per_seed:
            self._section("PER-SEED DETAIL")
            self._put(
                f"  {'Seed':>4s}  {'Acc(f)':>7s} {'Acc(m)':>7s}  "
                f"{'ECE(f)':>7s} {'ECE(m)':>7s}  "
                f"{'MCE(f)':>7s} {'MCE(m)':>7s}\n",
                'header',
            )
            self._put("  " + "─" * 56 + "\n", 'dim')

            import pandas as _pd
            df = _pd.DataFrame(per_seed)
            full_rows = df[df['variant'] == 'full']
            model_rows = df[df['variant'] == 'model']

            for seed in seeds:
                f_row = full_rows[full_rows['seed'] == seed]
                m_row = model_rows[model_rows['seed'] == seed]
                if f_row.empty or m_row.empty:
                    continue
                f = f_row.iloc[0]
                m = m_row.iloc[0]
                self._put(
                    f"  {seed:4d}  {f['accuracy']:.4f}  {m['accuracy']:.4f}  "
                    f"{f['ece']:.4f}  {m['ece']:.4f}  "
                    f"{f['mce']:.4f}  {m['mce']:.4f}\n",
                    'neutral',
                )

        self.text.config(state=tk.DISABLED)

    # ── Text helpers ─────────────────────────────────────────────

    def _section(self, title: str):
        self._put(f"\n  ── {title} ──\n", 'header')

    def _put(self, text: str, tag: str = 'neutral'):
        self.text.insert(tk.END, text, tag)
