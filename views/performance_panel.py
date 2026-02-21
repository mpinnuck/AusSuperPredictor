"""
Performance Panel - displays prediction history analytics
"""
import tkinter as tk
from tkinter import ttk, scrolledtext


class PerformancePanel(tk.Frame):
    """A panel that shows prediction performance analytics."""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self._build_ui()

    # ── UI construction ──────────────────────────────────────────

    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Top bar with refresh button
        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, sticky='ew', pady=(0, 5))
        self.refresh_btn = ttk.Button(bar, text="↻ Refresh", command=self._on_refresh)
        self.refresh_btn.pack(side=tk.LEFT, padx=5)
        self.status_label = ttk.Label(bar, text="", foreground='grey')
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Scrollable content area
        self.text = scrolledtext.ScrolledText(
            self, wrap=tk.WORD, font=('Courier', 11), state=tk.DISABLED
        )
        self.text.grid(row=1, column=0, sticky='nsew')

        # Colour tags
        self.text.tag_config('header', foreground='#1a1a2e', font=('Courier', 12, 'bold'))
        self.text.tag_config('good', foreground='green')
        self.text.tag_config('bad', foreground='red')
        self.text.tag_config('neutral', foreground='black')
        self.text.tag_config('dim', foreground='grey')
        self.text.tag_config('warn', foreground='orange')

        # Callback set by the window
        self._refresh_callback = None

    def set_refresh_callback(self, fn):
        """Set the function to call when Refresh is clicked.  fn() -> dict or None"""
        self._refresh_callback = fn

    # ── Public render method ─────────────────────────────────────

    def render(self, perf: dict | None, thresholds=None, drift: bool = False, perf_log=None):
        """Render the full performance dashboard.

        Args:
            perf: dict from DataManager.get_prediction_performance() or None
            thresholds: DataFrame from analyze_thresholds() or None
            drift: True if model drift was detected
            perf_log: DataFrame from get_performance_log() or None
        """
        self.text.config(state=tk.NORMAL)
        self.text.delete('1.0', tk.END)

        if perf is None:
            self._put("Not enough completed predictions yet (need ≥ 5).\n", 'dim')
            self._put("Run Predict daily and results will accumulate here.", 'dim')
            self.text.config(state=tk.DISABLED)
            self.status_label.config(text="No data")
            return

        # ── Overall ──────────────────────────────────────────────
        self._section("OVERALL ACCURACY")
        pct = perf['accuracy']
        tag = 'good' if pct >= 0.55 else 'bad' if pct < 0.50 else 'neutral'
        self._put(f"  {perf['correct']} / {perf['total']}  =  {pct:.1%}\n", tag)

        if perf.get('recent_10') is not None:
            r10 = perf['recent_10']
            tag10 = 'good' if r10 >= 0.55 else 'bad' if r10 < 0.50 else 'neutral'
            self._put(f"  Last 10 predictions:  {r10:.1%}\n", tag10)
        if perf.get('recent_20') is not None:
            r20 = perf['recent_20']
            tag20 = 'good' if r20 >= 0.55 else 'bad' if r20 < 0.50 else 'neutral'
            self._put(f"  Last 20 predictions:  {r20:.1%}\n", tag20)

        # ── Drift warning ────────────────────────────────────────
        if drift:
            self._put("\n  ⚠ MODEL DRIFT DETECTED — consider retraining\n", 'warn')

        # ── Hypothetical returns ─────────────────────────────────
        hyp = perf.get('hypothetical', {})
        if hyp:
            self._section("HYPOTHETICAL RETURNS")
            cum = hyp.get('cumulative_pct', 0)
            tag = 'good' if cum > 0 else 'bad' if cum < 0 else 'neutral'
            self._put(f"  Cumulative return:  {cum:>+.2%}\n", tag)
            self._put(f"  Mean per trade:     {hyp.get('mean_return', 0):>+.4%}\n", tag)
            self._put(f"  Trades taken:       {hyp.get('n_trades', 0)}\n", 'neutral')
            wr = hyp.get('win_rate', 0)
            wr_tag = 'good' if wr >= 0.55 else 'bad' if wr < 0.50 else 'neutral'
            self._put(f"  Win rate:           {wr:.1%}\n", wr_tag)
            self._put(f"  Avg win:            {hyp.get('avg_win', 0):>+.4%}\n", 'good')
            self._put(f"  Avg loss:           {hyp.get('avg_loss', 0):>+.4%}\n", 'bad')

        # ── By confidence level ──────────────────────────────────
        by_conf = perf.get('by_confidence', {})
        if by_conf:
            self._section("BY CONFIDENCE LEVEL")
            self._put(f"  {'Level':<14} {'Accuracy':>8}  {'Count':>5}\n", 'dim')
            self._put(f"  {'─'*14} {'─'*8}  {'─'*5}\n", 'dim')
            for level, stats in by_conf.items():
                acc = stats['accuracy']
                n = int(stats['n'])
                tag = 'good' if acc >= 0.60 else 'bad' if acc < 0.50 else 'neutral'
                self._put(f"  {level:<14} {acc:>7.1%}  {n:>5}\n", tag)

        # ── By day of week ───────────────────────────────────────
        by_day = perf.get('by_day', {})
        if by_day:
            self._section("BY DAY OF WEEK")
            self._put(f"  {'Day':<12} {'Accuracy':>8}  {'Count':>5}\n", 'dim')
            self._put(f"  {'─'*12} {'─'*8}  {'─'*5}\n", 'dim')
            for day, stats in by_day.items():
                acc = stats['accuracy']
                n = int(stats['n'])
                tag = 'good' if acc >= 0.60 else 'bad' if acc < 0.50 else 'neutral'
                self._put(f"  {day:<12} {acc:>7.1%}  {n:>5}\n", tag)

        # ── By market regime ─────────────────────────────────────
        by_regime = perf.get('by_regime', {})
        if by_regime:
            self._section("BY MARKET REGIME")
            self._put(f"  {'Regime':<16} {'Accuracy':>8}  {'Count':>5}\n", 'dim')
            self._put(f"  {'─'*16} {'─'*8}  {'─'*5}\n", 'dim')
            for regime, stats in by_regime.items():
                acc = stats['accuracy']
                n = int(stats['n'])
                tag = 'good' if acc >= 0.60 else 'bad' if acc < 0.50 else 'neutral'
                self._put(f"  {regime:<16} {acc:>7.1%}  {n:>5}\n", tag)

        # ── By model version ─────────────────────────────────────
        by_ver = perf.get('by_model_version', {})
        if by_ver:
            self._section("BY MODEL VERSION")
            self._put(f"  {'Version':<20} {'Accuracy':>8}  {'Count':>5}\n", 'dim')
            self._put(f"  {'─'*20} {'─'*8}  {'─'*5}\n", 'dim')
            for ver, stats in by_ver.items():
                acc = stats['accuracy']
                n = int(stats['n'])
                tag = 'good' if acc >= 0.60 else 'bad' if acc < 0.50 else 'neutral'
                self._put(f"  {str(ver):<20} {acc:>7.1%}  {n:>5}\n", tag)

        # ── Threshold analysis ───────────────────────────────────
        if thresholds is not None and not thresholds.empty:
            self._section("THRESHOLD ANALYSIS")
            self._put(f"  {'Threshold':>9}  {'Accuracy':>8}  {'Trades':>6}  {'Correct':>7}\n", 'dim')
            self._put(f"  {'─'*9}  {'─'*8}  {'─'*6}  {'─'*7}\n", 'dim')
            for _, row in thresholds.iterrows():
                acc = row['accuracy']
                tag = 'good' if acc >= 0.60 else 'bad' if acc < 0.50 else 'neutral'
                self._put(
                    f"  {row['threshold']:>9.2f}  {acc:>7.1%}  "
                    f"{int(row['trades']):>6}  {int(row['correct']):>7}\n",
                    tag,
                )

        # ── Accuracy trend ────────────────────────────────────────
        if perf_log is not None and len(perf_log) > 1:
            self._section("ACCURACY TREND")
            self._render_accuracy_trend(perf_log)

        # ── Recent predictions ───────────────────────────────────
        self._section("RECENT PREDICTIONS")
        self._render_recent_predictions()

        self.text.config(state=tk.DISABLED)
        self.status_label.config(
            text=f"{perf['total']} predictions  •  {perf['accuracy']:.1%} accuracy"
        )

    # ── Helpers ──────────────────────────────────────────────────

    def _section(self, title: str):
        self._put(f"\n  ── {title} {'─' * (40 - len(title))}\n\n", 'header')

    def _put(self, text: str, tag: str = 'neutral'):
        self.text.insert(tk.END, text, tag)

    def _render_recent_predictions(self):
        """Show last 10 predictions from the history CSV."""
        import os
        import pandas as pd

        path = 'data/asx200history.csv'
        if not os.path.exists(path):
            self._put("  No history file found.\n", 'dim')
            return

        hist = pd.read_csv(path)
        if hist.empty:
            self._put("  No predictions yet.\n", 'dim')
            return

        cols = ['prediction_date', 'probability', 'signal', 'confidence_level',
                'market_regime', 'actual_close', 'actual_return', 'success', 'result_label']
        show_cols = [c for c in cols if c in hist.columns]
        recent = hist[show_cols].tail(10).iloc[::-1]  # most recent first

        self._put(f"  {'Date':<12} {'Prob':>6} {'Decision':<20} {'Regime':<14} "
                  f"{'Return':>8} {'Result':<12}\n", 'dim')
        self._put(f"  {'─'*12} {'─'*6} {'─'*20} {'─'*14} "
                  f"{'─'*8} {'─'*12}\n", 'dim')

        for _, row in recent.iterrows():
            pred_date = str(row.get('prediction_date', ''))[:10]
            prob = row.get('probability', float('nan'))
            signal = str(row.get('signal', ''))
            regime = str(row.get('market_regime', ''))
            ret = row.get('actual_return', float('nan'))
            success = row.get('success', float('nan'))
            result_label = str(row.get('result_label', ''))

            prob_str = f"{prob*100:.0f}%" if not _isnan(prob) else '  —'
            ret_str = f"{ret*100:>+7.2f}%" if not _isnan(ret) else '      —'

            if _isnan(success):
                result_str = '…'
                tag = 'dim'
            elif 'CORRECT' in result_label:
                result_str = result_label
                tag = 'good'
            else:
                result_str = result_label if result_label and result_label != 'nan' else ('✓' if int(success) == 1 else '✗')
                tag = 'bad' if not _isnan(success) and int(success) == 0 else 'good'

            self._put(
                f"  {pred_date:<12} {prob_str:>6} {signal:<20} {regime:<14} "
                f"{ret_str} {result_str:<12}\n",
                tag,
            )

    def _render_accuracy_trend(self, perf_log):
        """Show how accuracy has evolved over time from performance_log.csv."""
        self._put(f"  {'Date':<12} {'Total':>6} {'Correct':>7} {'Accuracy':>8} "
                  f"{'Last 10':>8} {'Last 20':>8} {'Drift':>6}\n", 'dim')
        self._put(f"  {'─'*12} {'─'*6} {'─'*7} {'─'*8} "
                  f"{'─'*8} {'─'*8} {'─'*6}\n", 'dim')

        for _, row in perf_log.iterrows():
            snap_date = str(row.get('snapshot_date', ''))[:10]
            total = int(row.get('total_predictions', 0))
            correct = int(row.get('correct', 0))
            acc = row.get('accuracy', 0.0)
            r10 = row.get('recent_10', float('nan'))
            r20 = row.get('recent_20', float('nan'))
            drift = row.get('drift_detected', False)

            r10_str = f"{r10:>7.1%}" if not _isnan(r10) else '     —'
            r20_str = f"{r20:>7.1%}" if not _isnan(r20) else '     —'
            drift_str = '  ⚠' if drift else '  —'

            tag = 'good' if acc >= 0.55 else 'bad' if acc < 0.50 else 'neutral'
            self._put(
                f"  {snap_date:<12} {total:>6} {correct:>7} {acc:>7.1%} "
                f"{r10_str} {r20_str} {drift_str}\n",
                tag,
            )

    def _on_refresh(self):
        if self._refresh_callback:
            self._refresh_callback()


def _isnan(v) -> bool:
    """Return True for NaN, None, or empty string."""
    if v is None:
        return True
    try:
        import math
        return math.isnan(float(v))
    except (ValueError, TypeError):
        return False
