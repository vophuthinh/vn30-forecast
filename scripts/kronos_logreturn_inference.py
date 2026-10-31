#!/usr/bin/env python3
"""
Kronos Log-Return Inference — VN30
====================================
Transform absolute OHLCV → log-return space trước khi đưa vào Kronos,
sau đó reconstruct predictions về absolute price.

Motivation:
  - Kronos được train trên CN/US data (giá 1-300 USD/CNY)
  - VN30 giá 10,000-230,000 VND → distribution shift
  - Log-returns normalise về ~N(0, 0.02) bất kể giá tuyệt đối

Usage:
  python scripts/kronos_logreturn_inference.py
  python scripts/kronos_logreturn_inference.py --tickers VCB HPG
  python scripts/kronos_logreturn_inference.py --pred_len 20 --samples 15

Output:
  results/predictions_logreturn/<TICKER>_lr_pred.csv  — absolute OHLCV predictions
  results/charts_logreturn/<TICKER>_lr_pred.png       — comparison chart
  results/charts_logreturn/lr_comparison.png          — scatter ZS vs LR MAPE
  results/analysis/logreturn_comparison.md            — full comparison table
"""

import sys
import os
import argparse
import warnings
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
KRONOS_DIR = BASE_DIR / "models" / "Kronos-repo"
PROCESSED  = BASE_DIR / "data" / "processed"
OUT_DIR    = BASE_DIR / "results" / "predictions_logreturn"
CHART_DIR  = BASE_DIR / "results" / "charts_logreturn"
ANALYSIS   = BASE_DIR / "results" / "analysis"
ZS_METRICS = BASE_DIR / "results" / "predictions" / "_metrics.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(KRONOS_DIR))

# ── Config ────────────────────────────────────────────────────────────────────
LOOKBACK     = 400   # ngày lịch sử đưa vào context
PRED_LEN     = 20    # ngày dự báo
SAMPLE_COUNT = 15    # Monte Carlo samples
TEMPERATURE  = 1.0
TOP_P        = 0.9
VOL_ROLL_WIN = 30    # rolling window cho volume normalisation

MODEL_ID     = "NeoQuasar/Kronos-small"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"

ALL_TICKERS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]


# ── Load Kronos ───────────────────────────────────────────────────────────────
def load_kronos(device: str = "cpu"):
    from model import Kronos, KronosTokenizer, KronosPredictor

    print(f"📥 Loading Kronos tokenizer from {TOKENIZER_ID} …")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID)

    print(f"📥 Loading Kronos model from {MODEL_ID} …")
    model = Kronos.from_pretrained(MODEL_ID)

    print(f"🔧 Device: {device}")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    return predictor


# ── Log-return transforms ─────────────────────────────────────────────────────
def to_log_returns(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """
    Transform absolute OHLCV → log-return space.

    OHLC: expressed as log(price / prev_close)
    Volume/Amount: log deviation from 30-day rolling mean

    Returns (lr_df, last_close, min_positive_close).
    First row is dropped (NaN from shift).
    """
    df = df.copy()

    # Guard: replace 0s / negatives so log is safe
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].clip(lower=1e-6)
    df["volume"] = df["volume"].clip(lower=0)
    df["amount"] = df["amount"].clip(lower=0)

    prev_close = df["close"].shift(1)

    lr = pd.DataFrame(index=df.index)
    lr["close"]  = np.log(df["close"]  / prev_close)
    lr["open"]   = np.log(df["open"]   / prev_close)
    lr["high"]   = np.log(df["high"]   / prev_close)
    lr["low"]    = np.log(df["low"]    / prev_close)

    # Volume: log ratio to rolling mean (avoid log(0) with clip)
    vol_mean = df["volume"].rolling(VOL_ROLL_WIN, min_periods=1).mean().clip(lower=1)
    amt_mean = df["amount"].rolling(VOL_ROLL_WIN, min_periods=1).mean().clip(lower=1)
    # For zero volume days: log(0/mean) → -inf; set to 0 (neutral)
    safe_vol = df["volume"].replace(0, np.nan)
    safe_amt = df["amount"].replace(0, np.nan)
    lr["volume"] = np.log(safe_vol / vol_mean).fillna(0.0)
    lr["amount"] = np.log(safe_amt / amt_mean).fillna(0.0)

    # Drop first row (NaN from shift)
    lr = lr.iloc[1:].copy()

    # Clip extreme values (prevent model from seeing ±∞)
    lr = lr.clip(-3.0, 3.0)

    # Replace any remaining NaN/Inf with 0
    lr = lr.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    last_close = df["close"].iloc[-1]
    min_positive_close = df["close"][df["close"] > 0].min()

    return lr, last_close, min_positive_close


def from_log_returns(
    pred_lr: pd.DataFrame,
    last_close: float,
    min_positive_close: float,
    vol_ref: float = 1e6,
    amt_ref: float = 1e10,
) -> pd.DataFrame:
    """
    Reconstruct absolute OHLCV prices from log-return predictions.

    close[t] = close[t-1] * exp(lr_close[t])         seed = last_known_close
    open[t]  ≈ close[t]  * exp(lr_open[t] - lr_close[t])   (both rel to prev close)
    high[t]  ≈ close[t]  * exp(lr_high[t] - lr_close[t])
    low[t]   ≈ close[t]  * exp(lr_low[t]  - lr_close[t])
    volume   ≈ vol_ref * exp(lr_volume[t])
    """
    result = pd.DataFrame(index=pred_lr.index)

    # Reconstruct close sequentially
    closes = []
    prev = last_close
    for lr_c in pred_lr["close"].values:
        new_c = prev * np.exp(lr_c)
        # Guard against negative / zero prices
        new_c = max(new_c, min_positive_close * 0.5)
        closes.append(new_c)
        prev = new_c

    closes = np.array(closes)
    result["close"] = closes

    # OHLC: lr_open/high/low are all relative to the same prev_close as lr_close
    # So open/high/low relative to the reconstructed close:
    #   lr_open = log(open/prev_close)  =>  open = prev_close * exp(lr_open)
    #   But we can also write: open = close * exp(lr_open - lr_close)
    lr_close_arr = pred_lr["close"].values
    lr_open_arr  = pred_lr["open"].values
    lr_high_arr  = pred_lr["high"].values
    lr_low_arr   = pred_lr["low"].values

    open_prices  = closes * np.exp(lr_open_arr  - lr_close_arr)
    high_prices  = closes * np.exp(lr_high_arr  - lr_close_arr)
    low_prices   = closes * np.exp(lr_low_arr   - lr_close_arr)

    # Clip negative prices
    open_prices  = np.clip(open_prices,  min_positive_close * 0.5, None)
    high_prices  = np.clip(high_prices,  min_positive_close * 0.5, None)
    low_prices   = np.clip(low_prices,   min_positive_close * 0.5, None)

    # Enforce high >= close >= low (model may not respect this in lr space)
    high_prices  = np.maximum(high_prices, closes)
    low_prices   = np.minimum(low_prices,  closes)
    open_prices  = np.clip(open_prices, low_prices, high_prices)

    result["open"]   = open_prices
    result["high"]   = high_prices
    result["low"]    = low_prices

    # Volume / amount: inverse of log-deviation from rolling mean
    # We use median volume of the lookback window as vol_ref
    result["volume"] = (np.exp(pred_lr["volume"].values) * vol_ref).astype(int)
    result["amount"] = result["close"] * result["volume"]

    return result


# ── Metrics ───────────────────────────────────────────────────────────────────
def calc_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """MAE, MAPE, Direction Accuracy."""
    a = actual.values.astype(float)
    p = predicted.values.astype(float)

    mae  = float(np.mean(np.abs(a - p)))
    mape = float(np.mean(np.abs((a - p) / np.clip(a, 1e-6, None))) * 100)

    actual_dir = np.sign(np.diff(a, prepend=a[0]))
    pred_dir   = np.sign(np.diff(p, prepend=p[0]))
    dir_acc    = float(np.mean(actual_dir == pred_dir) * 100)

    return {"mae": round(mae, 2), "mape": round(mape, 4), "dir_acc": round(dir_acc, 2)}


# ── Plot per ticker ───────────────────────────────────────────────────────────
def plot_ticker(
    ticker: str,
    hist_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    actual_df: pd.DataFrame | None = None,
):
    tail = hist_df.tail(60).copy()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=False,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(
        f"{ticker}  —  Kronos log-return forecast ({PRED_LEN}D)",
        fontsize=14, fontweight="bold",
    )

    ax1.plot(range(len(tail)), tail["close"], color="#2196F3",
             linewidth=1.5, label="History (60d)")
    offset = len(tail)
    ax1.plot(
        range(offset - 1, offset + len(pred_df)),
        [tail["close"].iloc[-1]] + list(pred_df["close"]),
        color="#F44336", linewidth=1.8, linestyle="--",
        label=f"LR forecast ({PRED_LEN}D)",
    )

    if actual_df is not None and len(actual_df) > 0:
        ax1.plot(
            range(offset - 1, offset + len(actual_df)),
            [tail["close"].iloc[-1]] + list(actual_df["close"]),
            color="#4CAF50", linewidth=1.5, label="Actual",
        )

    ax1.set_ylabel("Close Price (VND)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    ax2.bar(range(len(tail)), tail["volume"] / 1e6, color="#90CAF9",
            alpha=0.7, label="History")
    ax2.bar(range(offset, offset + len(pred_df)), pred_df["volume"] / 1e6,
            color="#EF9A9A", alpha=0.7, label="Forecast")
    ax2.set_ylabel("Volume (M)", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = CHART_DIR / f"{ticker}_lr_pred.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ── Comparison scatter plot ───────────────────────────────────────────────────
def plot_comparison(compare_df: pd.DataFrame):
    """
    Scatter: x = ZS MAPE, y = LR MAPE
    Colour: direction accuracy improvement (LR DirAcc - ZS DirAcc)
    Diagonal: y=x  (above → LR worse, below → LR better)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    delta_dir = compare_df["lr_dir"] - compare_df["zs_dir"]
    sc = ax.scatter(
        compare_df["zs_mape"],
        compare_df["lr_mape"],
        c=delta_dir,
        cmap="RdYlGn",
        s=100,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.5,
        zorder=3,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("ΔDirAcc (LR − ZS, pp)", fontsize=10)

    # Diagonal y=x
    lim_max = max(compare_df["zs_mape"].max(), compare_df["lr_mape"].max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=1, alpha=0.5, label="y=x (equal)")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)

    # Label outliers (top 5 + bottom 5 by ΔMAPE)
    delta_mape = compare_df["lr_mape"] - compare_df["zs_mape"]
    outlier_idx = (
        list(delta_mape.nsmallest(5).index) +
        list(delta_mape.nlargest(5).index)
    )
    outlier_idx = list(set(outlier_idx))
    for i in outlier_idx:
        row = compare_df.loc[i]
        ax.annotate(
            row["ticker"],
            (row["zs_mape"], row["lr_mape"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            fontweight="bold",
        )

    # Shade regions
    ax.fill_between([0, lim_max], [0, lim_max], lim_max,
                    alpha=0.04, color="red", label="LR worse (above diagonal)")
    ax.fill_between([0, lim_max], 0, [0, lim_max],
                    alpha=0.04, color="green", label="LR better (below diagonal)")

    ax.set_xlabel("Zero-Shot MAPE (%)", fontsize=12)
    ax.set_ylabel("Log-Return MAPE (%)", fontsize=12)
    ax.set_title("VN30 — Zero-Shot vs Log-Return MAPE Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # Summary stats box
    n_better = (delta_mape < 0).sum()
    avg_delta = delta_mape.mean()
    textbox = (
        f"LR better: {n_better}/{len(compare_df)} tickers\n"
        f"Avg ΔMAPE: {avg_delta:+.2f}%"
    )
    ax.text(
        0.03, 0.97, textbox,
        transform=ax.transAxes,
        fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    out_path = CHART_DIR / "lr_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Comparison scatter → {out_path}")
    return out_path


# ── Inference per ticker ──────────────────────────────────────────────────────
def run_ticker(predictor, ticker: str, verbose: bool = False) -> dict | None:
    csv_path = PROCESSED / f"{ticker}.csv"
    if not csv_path.exists():
        print(f"  ❌ {ticker}: file not found ({csv_path})")
        return None

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    required_cols = ["open", "high", "low", "close", "volume", "amount"]
    for col in required_cols:
        if col not in df.columns:
            print(f"  ❌ {ticker}: missing column '{col}'")
            return None

    if len(df) < LOOKBACK + PRED_LEN + 2:   # +2 for shift loss
        print(f"  ⚠️  {ticker}: not enough data ({len(df)} < {LOOKBACK + PRED_LEN + 2})")
        return None

    # ── Split raw data ────────────────────────────────────────────────────────
    split_idx  = len(df) - PRED_LEN
    # Take lookback + 1 extra row so that after shift-drop we still have LOOKBACK rows
    raw_context = df.loc[
        split_idx - LOOKBACK - 1 : split_idx - 1,
        required_cols
    ].copy().reset_index(drop=True)

    actual_df = df.loc[split_idx : split_idx + PRED_LEN - 1].reset_index(drop=True)

    # Median volume of context window — used as vol_ref for reconstruction
    vol_ref = raw_context["volume"].median()
    if vol_ref <= 0:
        vol_ref = 1e6

    # ── Log-return transform ──────────────────────────────────────────────────
    try:
        lr_df, last_close, min_pos_close = to_log_returns(raw_context)
    except Exception as e:
        print(f"  ❌ {ticker}: log-return transform failed — {e}")
        return None

    if len(lr_df) < 10:
        print(f"  ⚠️  {ticker}: lr_df too short after transform ({len(lr_df)} rows)")
        return None

    # Trim to exactly LOOKBACK rows
    lr_context = lr_df.tail(LOOKBACK).copy().reset_index(drop=True)

    # Timestamps
    x_timestamp = (
        df.loc[split_idx - LOOKBACK : split_idx - 1, "date"]
        .reset_index(drop=True)
        .iloc[: len(lr_context)]
    )

    last_date    = df["date"].iloc[split_idx - 1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=PRED_LEN)
    y_timestamp  = pd.Series(future_dates[:PRED_LEN])

    # ── Kronos inference in log-return space ──────────────────────────────────
    try:
        pred_lr = predictor.predict(
            df=lr_context,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=PRED_LEN,
            T=TEMPERATURE,
            top_p=TOP_P,
            sample_count=SAMPLE_COUNT,
            verbose=verbose,
        )
    except Exception as e:
        print(f"  ❌ {ticker}: Kronos inference failed — {e}")
        return None

    # ── Reconstruct absolute prices ───────────────────────────────────────────
    try:
        pred_abs = from_log_returns(pred_lr, last_close, min_pos_close, vol_ref=vol_ref)
    except Exception as e:
        print(f"  ❌ {ticker}: reconstruction failed — {e}")
        return None

    # ── Save predictions ──────────────────────────────────────────────────────
    pred_abs["date"] = future_dates[:PRED_LEN]
    pred_out = OUT_DIR / f"{ticker}_lr_pred.csv"
    pred_abs.to_csv(pred_out, index=False)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = calc_metrics(actual_df["close"], pred_abs["close"])

    # ── Chart ─────────────────────────────────────────────────────────────────
    try:
        plot_ticker(
            ticker,
            hist_df=df.loc[:split_idx - 1],
            pred_df=pred_abs,
            actual_df=actual_df,
        )
    except Exception as e:
        print(f"  ⚠️  {ticker}: chart failed — {e}")

    print(
        f"  ✅ {ticker:5s}  "
        f"MAE={metrics['mae']:>10,.0f}  "
        f"MAPE={metrics['mape']:6.2f}%  "
        f"DirAcc={metrics['dir_acc']:5.1f}%"
    )

    return {"ticker": ticker, **metrics}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global PRED_LEN, SAMPLE_COUNT

    parser = argparse.ArgumentParser(description="Kronos VN30 Log-Return Inference")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Danh sách mã (mặc định: tất cả 30 mã)")
    parser.add_argument("--pred_len", type=int, default=PRED_LEN)
    parser.add_argument("--samples",  type=int, default=SAMPLE_COUNT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    import torch

    PRED_LEN     = args.pred_len
    SAMPLE_COUNT = args.samples
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tickers = args.tickers if args.tickers else ALL_TICKERS

    print(f"\n{'='*62}")
    print(f"  Kronos VN30 — Log-Return Normalization Inference")
    print(f"  Tickers : {', '.join(tickers)}")
    print(f"  Lookback={LOOKBACK}D  Pred={PRED_LEN}D  Samples={SAMPLE_COUNT}")
    print(f"  Device  : {device}")
    print(f"{'='*62}\n")

    predictor = load_kronos(device=device)
    print()

    results = []
    for ticker in tickers:
        row = run_ticker(predictor, ticker, verbose=args.verbose)
        if row:
            results.append(row)

    if not results:
        print("\n❌ Không có kết quả nào thành công.")
        return

    lr_metrics_df = pd.DataFrame(results)
    lr_metrics_path = OUT_DIR / "_metrics_lr.csv"
    lr_metrics_df.to_csv(lr_metrics_path, index=False)

    print(f"\n{'='*62}")
    print(f"  TỔNG KẾT LOG-RETURN — {len(results)}/{len(tickers)} mã thành công")
    print(f"  Avg MAPE  : {lr_metrics_df['mape'].mean():.2f}%")
    print(f"  Avg DirAcc: {lr_metrics_df['dir_acc'].mean():.1f}%")
    print(f"{'='*62}\n")

    # ── Load zero-shot metrics for comparison ─────────────────────────────────
    if ZS_METRICS.exists():
        zs_df = pd.read_csv(ZS_METRICS)
        zs_df = zs_df.rename(columns={
            "mae":     "zs_mae",
            "mape":    "zs_mape",
            "dir_acc": "zs_dir",
        })

        compare = lr_metrics_df.rename(columns={
            "mae":     "lr_mae",
            "mape":    "lr_mape",
            "dir_acc": "lr_dir",
        }).merge(zs_df, on="ticker", how="inner")

        compare["delta_mape"] = compare["lr_mape"] - compare["zs_mape"]
        compare["delta_dir"]  = compare["lr_dir"]  - compare["zs_dir"]

        # ── Summary stats ─────────────────────────────────────────────────────
        avg_zs_mape  = compare["zs_mape"].mean()
        avg_lr_mape  = compare["lr_mape"].mean()
        avg_zs_dir   = compare["zs_dir"].mean()
        avg_lr_dir   = compare["lr_dir"].mean()

        top5_improve = compare.nsmallest(5, "delta_mape")[
            ["ticker", "zs_mape", "lr_mape", "delta_mape", "zs_dir", "lr_dir", "delta_dir"]
        ]
        bot5_worse = compare.nlargest(5, "delta_mape")[
            ["ticker", "zs_mape", "lr_mape", "delta_mape", "zs_dir", "lr_dir", "delta_dir"]
        ]

        print("  ┌─ COMPARISON ─────────────────────────────────────────┐")
        print(f"  │  ZS  avg MAPE : {avg_zs_mape:6.2f}%   DirAcc: {avg_zs_dir:5.1f}%        │")
        print(f"  │  LR  avg MAPE : {avg_lr_mape:6.2f}%   DirAcc: {avg_lr_dir:5.1f}%        │")
        print(f"  │  ΔMAPE        : {avg_lr_mape - avg_zs_mape:+6.2f}%                         │")
        print(f"  │  ΔDirAcc      : {avg_lr_dir  - avg_zs_dir:+6.1f}pp                        │")
        n_better = (compare["delta_mape"] < 0).sum()
        print(f"  │  LR better    : {n_better}/{len(compare)} tickers                   │")
        print("  └──────────────────────────────────────────────────────┘")

        print("\n  🏆 Top-5 cải thiện nhất (ΔMAPE âm = tốt hơn):")
        print(top5_improve.to_string(index=False))
        print("\n  ⚠️  Bottom-5 tệ hơn sau log-return:")
        print(bot5_worse.to_string(index=False))

        # ── Comparison scatter ────────────────────────────────────────────────
        plot_comparison(compare)

        # ── Markdown comparison table ─────────────────────────────────────────
        md_path = ANALYSIS / "logreturn_comparison.md"
        _write_markdown_report(compare, md_path,
                               avg_zs_mape, avg_lr_mape, avg_zs_dir, avg_lr_dir,
                               top5_improve, bot5_worse)

    else:
        print(f"\n  ℹ️  Zero-shot metrics not found at {ZS_METRICS}.")
        print("  Run scripts/kronos_inference.py first for comparison.")


# ── Markdown report ───────────────────────────────────────────────────────────
def _write_markdown_report(
    compare: pd.DataFrame,
    path: Path,
    avg_zs_mape: float, avg_lr_mape: float,
    avg_zs_dir: float,  avg_lr_dir: float,
    top5: pd.DataFrame, bot5: pd.DataFrame,
):
    delta_mape_avg = avg_lr_mape - avg_zs_mape
    delta_dir_avg  = avg_lr_dir  - avg_zs_dir
    n_better       = (compare["delta_mape"] < 0).sum()
    conclusion_icon = "✅" if delta_mape_avg < 0 else "❌"

    lines = [
        "# VN30 — Log-Return vs Zero-Shot: Comparison Report",
        "",
        f"> Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Summary",
        "",
        "| Metric | Zero-Shot | Log-Return | Δ |",
        "|--------|-----------|------------|---|",
        f"| Avg MAPE | {avg_zs_mape:.2f}% | {avg_lr_mape:.2f}% | {delta_mape_avg:+.2f}% |",
        f"| Avg DirAcc | {avg_zs_dir:.1f}% | {avg_lr_dir:.1f}% | {delta_dir_avg:+.1f}pp |",
        f"| # tickers LR better (MAPE) | — | {n_better}/{len(compare)} | — |",
        "",
        "## Conclusion",
        "",
    ]

    if delta_mape_avg < -1:
        lines.append(
            f"{conclusion_icon} **Log-return normalization giúp ích đáng kể**: "
            f"MAPE giảm trung bình {abs(delta_mape_avg):.2f}% trên toàn bộ VN30. "
            "Distribution shift được giảm thiểu rõ ràng."
        )
    elif delta_mape_avg < 0:
        lines.append(
            f"{conclusion_icon} **Log-return normalization có cải thiện nhẹ**: "
            f"MAPE giảm trung bình {abs(delta_mape_avg):.2f}%. "
            "Cải thiện khiêm tốn, nhưng có thể kết hợp với fine-tuning."
        )
    else:
        lines.append(
            f"{conclusion_icon} **Log-return normalization KHÔNG cải thiện**: "
            f"MAPE tăng trung bình {delta_mape_avg:.2f}%. "
            "Kronos có thể đã handle price magnitude tốt ở tokenizer level. "
            "Khuyến nghị: thử fine-tune trực tiếp trên absolute prices."
        )

    lines += [
        "",
        "## Full Comparison Table",
        "",
        "| Ticker | ZS MAPE | LR MAPE | ΔMAPE | ZS Dir | LR Dir | ΔDir |",
        "|--------|---------|---------|-------|--------|--------|------|",
    ]

    for _, row in compare.sort_values("delta_mape").iterrows():
        mape_icon = "✅" if row["delta_mape"] < 0 else "❌"
        dir_icon  = "✅" if row["delta_dir"]  > 0 else ("➖" if row["delta_dir"] == 0 else "❌")
        lines.append(
            f"| {row['ticker']} | {row['zs_mape']:.2f}% | {row['lr_mape']:.2f}% "
            f"| {mape_icon} {row['delta_mape']:+.2f}% "
            f"| {row['zs_dir']:.0f}% | {row['lr_dir']:.0f}% "
            f"| {dir_icon} {row['delta_dir']:+.0f}pp |"
        )

    lines += [
        "",
        "## Top-5 Most Improved",
        "",
        "| Ticker | ZS MAPE | LR MAPE | ΔMAPE | ΔDir |",
        "|--------|---------|---------|-------|------|",
    ]
    for _, row in top5.iterrows():
        lines.append(
            f"| {row['ticker']} | {row['zs_mape']:.2f}% | {row['lr_mape']:.2f}% "
            f"| {row['delta_mape']:+.2f}% | {row['delta_dir']:+.0f}pp |"
        )

    lines += [
        "",
        "## Bottom-5 Most Degraded",
        "",
        "| Ticker | ZS MAPE | LR MAPE | ΔMAPE | ΔDir |",
        "|--------|---------|---------|-------|------|",
    ]
    for _, row in bot5.iterrows():
        lines.append(
            f"| {row['ticker']} | {row['zs_mape']:.2f}% | {row['lr_mape']:.2f}% "
            f"| {row['delta_mape']:+.2f}% | {row['delta_dir']:+.0f}pp |"
        )

    lines += [
        "",
        "---",
        "*Chart: `results/charts_logreturn/lr_comparison.png`*",
        "*Predictions: `results/predictions_logreturn/`*",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  📝 Markdown report → {path}")


if __name__ == "__main__":
    main()
