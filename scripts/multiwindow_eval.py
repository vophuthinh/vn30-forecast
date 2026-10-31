#!/usr/bin/env python3
"""LEGACY: evaluates a single 20-day window (Oct 2026) or 5 windows (Jun-Oct 2026) with no baselines.
Results are noise-level; use scripts/rolling_eval.py + scripts/rolling_report.py instead.

Multi-Window Walk-Forward Evaluation — Kronos VN30
====================================================
Đánh giá Kronos Zero-Shot trên 5 windows × 30 mã = 150 predictions.

Windows (non-overlapping, từ cuối lùi dần):
  W5 (newest): df[-20:]        — Oct 2026
  W4:          df[-40:-20]     — Sep 2026
  W3:          df[-60:-40]     — Aug 2026
  W2:          df[-80:-60]     — Jul 2026
  W1:          df[-100:-80]    — Jun 2026

Context cho mỗi window: 400 ngày trước window đó (không overlap).

Output:
  results/multiwindow/per_window_metrics.csv    — ticker × window × metrics
  results/multiwindow/aggregated_metrics.csv    — ticker × (mean/std per metric)
  results/multiwindow/reliable_tickers.txt      — mean_dir > 55% & std_dir < 10%
  results/charts/multiwindow_heatmap.png        — heatmap DirAcc

Usage:
  python scripts/multiwindow_eval.py
  python scripts/multiwindow_eval.py --tickers VCB HPG FPT
"""

import sys
import os
import argparse
import warnings
from pathlib import Path
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")   # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
KRONOS_DIR  = BASE_DIR / "models" / "Kronos-repo"
PROCESSED   = BASE_DIR / "data" / "processed"
MW_OUT      = BASE_DIR / "results" / "multiwindow"
CHART_OUT   = BASE_DIR / "results" / "charts"

MW_OUT.mkdir(parents=True, exist_ok=True)
CHART_OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(KRONOS_DIR))

# ── Config ───────────────────────────────────────────────────────────────────
LOOKBACK      = 400    # context days before each window
PRED_LEN      = 20     # days per window
SAMPLE_COUNT  = 10     # Monte Carlo samples (fast)
TEMPERATURE   = 1.0
TOP_P         = 0.9
N_WINDOWS     = 5

MODEL_ID      = "NeoQuasar/Kronos-small"
TOKENIZER_ID  = "NeoQuasar/Kronos-Tokenizer-base"

ALL_TICKERS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]

# Window definitions: (start_offset_from_end, end_offset_from_end)
# Positive offsets counted from end: df[-end:-start] or df[-end:] for newest
WINDOWS = [
    ("W1", -100, -80),   # df[-100:-80]  Jun 2026
    ("W2",  -80, -60),   # df[-80:-60]   Jul 2026
    ("W3",  -60, -40),   # df[-60:-40]   Aug 2026
    ("W4",  -40, -20),   # df[-40:-20]   Sep 2026
    ("W5",  -20, None),  # df[-20:]      Oct 2026
]


# ── Load Kronos ───────────────────────────────────────────────────────────────
def load_kronos(device: str = "cpu"):
    from model import Kronos, KronosTokenizer, KronosPredictor

    print(f"📥 Loading tokenizer: {TOKENIZER_ID}")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID)

    print(f"📥 Loading model: {MODEL_ID}")
    model = Kronos.from_pretrained(MODEL_ID)

    print(f"🔧 Device: {device}")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    return predictor


# ── Metrics ───────────────────────────────────────────────────────────────────
def calc_metrics(actual: np.ndarray, predicted: np.ndarray, last_context_price: float) -> dict:
    """Compute MAE, MAPE, Direction Accuracy (Hit Rate)."""
    mae  = float(np.mean(np.abs(actual - predicted)))
    # avoid div-by-zero
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100)

    # Direction accuracy: compare consecutive closes
    # Prepend last context close so index 0 gets a valid diff
    actual_ext = np.concatenate([[last_context_price], actual])
    pred_ext   = np.concatenate([[last_context_price], predicted])

    actual_dir = np.sign(np.diff(actual_ext))
    pred_dir   = np.sign(np.diff(pred_ext))

    dir_acc = float(np.mean(actual_dir == pred_dir) * 100)

    return {
        "mae":     round(mae, 2),
        "mape":    round(mape, 4),
        "dir_acc": round(dir_acc, 2),
    }


# ── Run one (ticker, window) pair ─────────────────────────────────────────────
def run_one(predictor, ticker: str, win_name: str, start_idx: int, end_idx,
            df: pd.DataFrame, verbose: bool = False) -> dict | None:
    """
    Run Kronos prediction for one ticker on one window slice.

    Parameters
    ----------
    start_idx, end_idx : integer positions into df (already resolved from negative offsets)
        target window = df[start_idx:end_idx]
        context       = df[start_idx - LOOKBACK : start_idx]
    """
    n = len(df)

    # Validate enough rows
    if start_idx < LOOKBACK:
        print(f"  ⚠️  {ticker}/{win_name}: not enough lookback rows ({start_idx} < {LOOKBACK})")
        return None

    target_slice  = df.iloc[start_idx:end_idx].reset_index(drop=True)
    context_slice = df.iloc[start_idx - LOOKBACK : start_idx].reset_index(drop=True)

    if len(target_slice) < PRED_LEN:
        print(f"  ⚠️  {ticker}/{win_name}: target window too short ({len(target_slice)} < {PRED_LEN})")
        return None

    feature_cols = ["open", "high", "low", "close", "volume", "amount"]
    x_df = context_slice[feature_cols].copy()

    # Timestamps
    x_timestamp = context_slice["date"].reset_index(drop=True)
    last_date   = pd.Timestamp(context_slice["date"].iloc[-1])
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=PRED_LEN)
    y_timestamp = pd.Series(future_dates[:PRED_LEN])

    actual_close         = target_slice["close"].values[:PRED_LEN]
    last_context_close   = context_slice["close"].iloc[-1]

    try:
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=PRED_LEN,
            T=TEMPERATURE,
            top_p=TOP_P,
            sample_count=SAMPLE_COUNT,
            verbose=verbose,
        )

        predicted_close = pred_df["close"].values[:PRED_LEN]
        metrics = calc_metrics(actual_close, predicted_close, last_context_close)

        print(f"  ✅ {ticker:5s}/{win_name}  "
              f"MAE={metrics['mae']:>12,.0f}  "
              f"MAPE={metrics['mape']:5.2f}%  "
              f"DirAcc={metrics['dir_acc']:5.1f}%")

        return {
            "ticker":   ticker,
            "window":   win_name,
            **metrics,
        }

    except Exception as e:
        print(f"  ❌ {ticker}/{win_name}: {e}")
        return None


# ── Heatmap ───────────────────────────────────────────────────────────────────
def plot_heatmap(pivot: pd.DataFrame, out_path: Path):
    """Heatmap: x=window, y=ticker, color=DirAcc (%)."""
    fig, ax = plt.subplots(figsize=(9, max(8, len(pivot) * 0.38)))

    # Colormap centered at 50% (random baseline)
    cmap = plt.get_cmap("RdYlGn")
    norm = mcolors.TwoSlopeNorm(vmin=30, vcenter=50, vmax=70)

    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    # Annotate each cell
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=7.5, color="black" if 40 < val < 60 else "white",
                        fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Direction Accuracy (%)", fontsize=10)

    ax.set_title("Kronos VN30 — Direction Accuracy per Window (Walk-Forward)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Window", fontsize=10)
    ax.set_ylabel("Ticker", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Heatmap saved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Kronos VN30 Multi-Window Walk-Forward Evaluation")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Subset of tickers (default: all 30)")
    parser.add_argument("--samples", type=int, default=SAMPLE_COUNT,
                        help=f"Monte Carlo samples per prediction (default: {SAMPLE_COUNT})")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output from Kronos predictor")
    args = parser.parse_args()

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tickers = args.tickers if args.tickers else ALL_TICKERS
    sample_count = args.samples

    print(f"\n{'='*65}")
    print(f"  Kronos VN30 — Multi-Window Walk-Forward Evaluation")
    print(f"  Tickers   : {len(tickers)} mã")
    print(f"  Windows   : {N_WINDOWS} × {PRED_LEN}D (non-overlapping)")
    print(f"  Lookback  : {LOOKBACK}D per window")
    print(f"  Samples   : {sample_count}")
    print(f"  Device    : {device}")
    print(f"{'='*65}\n")

    # ── Load model once ────────────────────────────────────────────────────
    predictor = load_kronos(device=device)
    print()

    # ── Loop over tickers × windows ───────────────────────────────────────
    all_results = []
    total = len(tickers) * N_WINDOWS
    done  = 0

    for ticker in tickers:
        csv_path = PROCESSED / f"{ticker}.csv"
        if not csv_path.exists():
            print(f"⚠️  {ticker}: CSV not found, skipping")
            continue

        df = pd.read_csv(csv_path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        n  = len(df)

        print(f"\n── {ticker} ({n} rows, last: {df['date'].iloc[-1].date()}) ──")

        for win_name, neg_start, neg_end in WINDOWS:
            # Resolve negative indices to absolute positions
            abs_start = n + neg_start                  # e.g. n-100
            abs_end   = (n + neg_end) if neg_end is not None else n  # e.g. n-80 or n

            result = run_one(
                predictor, ticker, win_name,
                start_idx=abs_start,
                end_idx=abs_end,
                df=df,
                verbose=args.verbose,
            )
            done += 1

            if result:
                all_results.append(result)

            print(f"  Progress: {done}/{total}", end="\r")

    print()

    if not all_results:
        print("❌ No results collected — exiting.")
        return

    # ── Per-window CSV ─────────────────────────────────────────────────────
    per_win_df = pd.DataFrame(all_results)
    per_win_path = MW_OUT / "per_window_metrics.csv"
    per_win_df.to_csv(per_win_path, index=False)
    print(f"\n📄 Per-window metrics → {per_win_path}")

    # ── Aggregated CSV (mean ± std per ticker) ─────────────────────────────
    agg = per_win_df.groupby("ticker").agg(
        mean_mae   =("mae",     "mean"),
        std_mae    =("mae",     "std"),
        mean_mape  =("mape",    "mean"),
        std_mape   =("mape",    "std"),
        mean_dir   =("dir_acc", "mean"),
        std_dir    =("dir_acc", "std"),
        n_windows  =("window",  "count"),
    ).round(4).reset_index()

    agg_path = MW_OUT / "aggregated_metrics.csv"
    agg.to_csv(agg_path, index=False)
    print(f"📄 Aggregated metrics → {agg_path}")

    # ── Reliable tickers ───────────────────────────────────────────────────
    reliable = agg[(agg["mean_dir"] > 55) & (agg["std_dir"] < 10)].copy()
    reliable = reliable.sort_values("mean_dir", ascending=False)
    reliable_path = MW_OUT / "reliable_tickers.txt"
    with open(reliable_path, "w") as f:
        f.write("# Reliable Tickers: mean DirAcc > 55% AND std DirAcc < 10%\n")
        f.write(f"# Generated: multi-window walk-forward ({N_WINDOWS} windows × {PRED_LEN}D)\n\n")
        if reliable.empty:
            f.write("(none met criteria)\n")
        else:
            f.write(f"{'Ticker':<8}  {'MeanDir':>8}  {'StdDir':>8}  {'MeanMAPE':>10}  {'Windows':>7}\n")
            f.write("-" * 50 + "\n")
            for _, row in reliable.iterrows():
                f.write(f"{row['ticker']:<8}  {row['mean_dir']:8.2f}  {row['std_dir']:8.2f}"
                        f"  {row['mean_mape']:10.4f}  {int(row['n_windows']):7d}\n")
    print(f"📄 Reliable tickers → {reliable_path}")

    # ── Heatmap ───────────────────────────────────────────────────────────
    pivot = per_win_df.pivot_table(index="ticker", columns="window", values="dir_acc")
    # Sort tickers by mean DirAcc desc
    pivot = pivot.reindex(agg.sort_values("mean_dir", ascending=False)["ticker"])
    # Ensure window columns in order W1..W5
    win_order = [w[0] for w in WINDOWS]
    pivot = pivot.reindex(columns=[c for c in win_order if c in pivot.columns])

    heatmap_path = CHART_OUT / "multiwindow_heatmap.png"
    plot_heatmap(pivot, heatmap_path)

    # ── Summary ───────────────────────────────────────────────────────────
    top10 = agg.sort_values("mean_dir", ascending=False).head(10)
    bot5  = agg.sort_values("mean_dir", ascending=True).head(5)
    beat_baseline = agg[(agg["mean_dir"] > 50)].shape[0]

    print(f"\n{'='*65}")
    print(f"  MULTI-WINDOW WALK-FORWARD SUMMARY")
    print(f"  Tickers evaluated : {agg.shape[0]}")
    print(f"  Total predictions : {len(all_results)}")
    print(f"  Avg MAPE          : {agg['mean_mape'].mean():.2f}% ± {agg['std_mape'].mean():.2f}%")
    print(f"  Avg DirAcc        : {agg['mean_dir'].mean():.1f}% ± {agg['std_dir'].mean():.1f}%")
    print(f"  Baseline (random) = 50%  →  {beat_baseline} mã beat baseline consistently")
    print(f"\n  Top 10 Reliable (mean DirAcc > 55%, std < 10%):")
    print(f"  {'Ticker':<8}  {'MeanDir':>8}  {'StdDir':>7}  {'MeanMAPE':>10}")
    print(f"  {'-'*42}")
    for _, row in top10.iterrows():
        marker = "⭐" if row["mean_dir"] > 55 and row["std_dir"] < 10 else "  "
        print(f"  {marker}{row['ticker']:<6}  {row['mean_dir']:8.2f}%  "
              f"{row['std_dir']:7.2f}%  {row['mean_mape']:10.4f}%")

    print(f"\n  Bottom 5 Unstable (lowest mean DirAcc):")
    print(f"  {'Ticker':<8}  {'MeanDir':>8}  {'StdDir':>7}  {'MeanMAPE':>10}")
    print(f"  {'-'*42}")
    for _, row in bot5.iterrows():
        print(f"  {'  '}{row['ticker']:<6}  {row['mean_dir']:8.2f}%  "
              f"{row['std_dir']:7.2f}%  {row['mean_mape']:10.4f}%")

    print(f"\n  Output files:")
    print(f"    {per_win_path}")
    print(f"    {agg_path}")
    print(f"    {reliable_path}")
    print(f"    {heatmap_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
