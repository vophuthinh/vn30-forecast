#!/usr/bin/env python3
"""
Kronos Zero-Shot Inference — VN30
===================================
Chạy dự báo giá cổ phiếu với Kronos-small (zero-shot, không cần train).

Usage:
  python scripts/kronos_inference.py              # dự báo tất cả 30 mã
  python scripts/kronos_inference.py --tickers VCB HPG FPT  # chỉ một số mã
  python scripts/kronos_inference.py --pred_len 20           # tùy pred length

Output:
  results/predictions/<TICKER>_pred.csv  — OHLCV dự báo
  results/predictions/_metrics.csv       — MAE, MAPE, Direction Accuracy
  results/charts/<TICKER>_pred.png       — biểu đồ close + volume
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
matplotlib.use("Agg")   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
KRONOS_DIR   = BASE_DIR / "models" / "Kronos-repo"
PROCESSED    = BASE_DIR / "data" / "processed"
PRED_OUT     = BASE_DIR / "results" / "predictions"
CHART_OUT    = BASE_DIR / "results" / "charts"

PRED_OUT.mkdir(parents=True, exist_ok=True)
CHART_OUT.mkdir(parents=True, exist_ok=True)

# Add Kronos source to path
sys.path.insert(0, str(KRONOS_DIR))

# ── Config ───────────────────────────────────────────────────────────────────
LOOKBACK     = 400     # số ngày lịch sử đưa vào context (≤512 cho Kronos-small)
PRED_LEN     = 20      # số ngày dự báo (~1 tháng giao dịch)
SAMPLE_COUNT = 20      # số Monte Carlo samples để lấy trung bình
TEMPERATURE  = 1.0     # sampling temperature
TOP_P        = 0.9     # nucleus sampling

MODEL_ID     = "NeoQuasar/Kronos-small"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"


# ── Load Kronos ───────────────────────────────────────────────────────────────
def load_kronos(device: str = "cpu"):
    from model import Kronos, KronosTokenizer, KronosPredictor

    print(f"📥 Loading Kronos tokenizer from {TOKENIZER_ID} ...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID)

    print(f"📥 Loading Kronos model from {MODEL_ID} ...")
    model = Kronos.from_pretrained(MODEL_ID)

    print(f"🔧 Device: {device}")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    return predictor


# ── Metrics ───────────────────────────────────────────────────────────────────
def calc_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """Tính MAE, MAPE, Direction Accuracy so với actual."""
    actual    = actual.values
    predicted = predicted.values

    mae  = float(np.mean(np.abs(actual - predicted)))
    mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100)

    # Direction: dấu thay đổi so với ngày cuối của context
    actual_dir = np.sign(np.diff(actual, prepend=actual[0]))
    pred_dir   = np.sign(np.diff(predicted, prepend=predicted[0]))
    dir_acc    = float(np.mean(actual_dir == pred_dir) * 100)

    return {"mae": round(mae, 2), "mape": round(mape, 4), "dir_acc": round(dir_acc, 2)}


# ── Plot ─────────────────────────────────────────────────────────────────────
def plot_result(ticker: str, hist_df: pd.DataFrame, pred_df: pd.DataFrame,
                actual_df: pd.DataFrame | None = None):
    """Vẽ biểu đồ Close + Volume: lịch sử | dự báo | actual (nếu có)."""
    tail = hist_df.tail(60).copy()     # 60 ngày gần nhất làm nền

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=False,
                                   gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"{ticker}  —  Kronos zero-shot forecast ({PRED_LEN}D)",
                 fontsize=14, fontweight="bold")

    # ── Close price ──
    ax1.plot(range(len(tail)), tail["close"], color="#2196F3",
             linewidth=1.5, label="Lịch sử (60 ngày)")
    offset = len(tail)
    ax1.plot(range(offset - 1, offset + len(pred_df)),
             [tail["close"].iloc[-1]] + list(pred_df["close"]),
             color="#F44336", linewidth=1.8, linestyle="--", label=f"Dự báo ({PRED_LEN}D)")

    if actual_df is not None and len(actual_df) > 0:
        ax1.plot(range(offset - 1, offset + len(actual_df)),
                 [tail["close"].iloc[-1]] + list(actual_df["close"]),
                 color="#4CAF50", linewidth=1.5, label="Thực tế")

    ax1.set_ylabel("Close Price (VND)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # ── Volume ──
    ax2.bar(range(len(tail)), tail["volume"] / 1e6, color="#90CAF9",
            alpha=0.7, label="Lịch sử")
    ax2.bar(range(offset, offset + len(pred_df)), pred_df["volume"] / 1e6,
            color="#EF9A9A", alpha=0.7, label="Dự báo")
    ax2.set_ylabel("Volume (triệu)", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = CHART_OUT / f"{ticker}_pred.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ── Inference per ticker ─────────────────────────────────────────────────────
def run_ticker(predictor, ticker: str, verbose: bool = False) -> dict | None:
    csv_path = PROCESSED / f"{ticker}.csv"
    if not csv_path.exists():
        print(f"  ❌ {ticker}: file not found ({csv_path})")
        return None

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < LOOKBACK + PRED_LEN:
        print(f"  ⚠️  {ticker}: not enough data ({len(df)} < {LOOKBACK + PRED_LEN})")
        return None

    # Split: context + holdout (dùng để tính metrics)
    split_idx  = len(df) - PRED_LEN
    x_df       = df.loc[split_idx - LOOKBACK : split_idx - 1,
                         ["open", "high", "low", "close", "volume", "amount"]].copy()
    x_df       = x_df.reset_index(drop=True)
    x_timestamp = df.loc[split_idx - LOOKBACK : split_idx - 1, "date"].reset_index(drop=True)

    # Future timestamps (generate business-day-like dates)
    last_date = df["date"].iloc[split_idx - 1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=PRED_LEN)
    y_timestamp = pd.Series(future_dates[:PRED_LEN])

    # Actual holdout
    actual_df = df.loc[split_idx : split_idx + PRED_LEN - 1].reset_index(drop=True)

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

        # Save prediction CSV
        pred_df["date"] = future_dates[:PRED_LEN]
        pred_out = PRED_OUT / f"{ticker}_pred.csv"
        pred_df.to_csv(pred_out, index=False)

        # Metrics
        metrics = calc_metrics(actual_df["close"], pred_df["close"])

        # Chart
        chart_path = plot_result(ticker,
                                  hist_df=df.loc[:split_idx - 1],
                                  pred_df=pred_df,
                                  actual_df=actual_df)

        print(f"  ✅ {ticker:5s}  "
              f"MAE={metrics['mae']:>10,.0f}  "
              f"MAPE={metrics['mape']:5.2f}%  "
              f"DirAcc={metrics['dir_acc']:5.1f}%  "
              f"→ {pred_out.name}")

        return {"ticker": ticker, **metrics}

    except Exception as e:
        print(f"  ❌ {ticker}: {e}")
        return None


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    global PRED_LEN, SAMPLE_COUNT

    parser = argparse.ArgumentParser(description="Kronos VN30 Zero-Shot Inference")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Danh sách mã cổ phiếu (mặc định: tất cả 30 mã)")
    parser.add_argument("--pred_len", type=int, default=PRED_LEN,
                        help=f"Số ngày dự báo (default: {PRED_LEN})")
    parser.add_argument("--samples", type=int, default=SAMPLE_COUNT,
                        help=f"Monte Carlo samples (default: {SAMPLE_COUNT})")
    parser.add_argument("--verbose", action="store_true",
                        help="In chi tiết từng bước inference")
    args = parser.parse_args()

    import torch

    pred_len     = args.pred_len
    sample_count = args.samples
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # All VN30 tickers
    all_tickers = [
        "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
        "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
        "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
    ]
    tickers = args.tickers if args.tickers else all_tickers

    print(f"\n{'='*60}")
    print(f"  Kronos VN30 — Zero-Shot Inference")
    print(f"  Mã: {', '.join(tickers)}")
    print(f"  Lookback={LOOKBACK}D  Pred={pred_len}D  Samples={sample_count}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Pass runtime args to module-level constants used by helper functions
    PRED_LEN     = pred_len
    SAMPLE_COUNT = sample_count

    # Load model ONCE, reuse for all tickers
    predictor = load_kronos(device=device)
    print()

    results = []
    for ticker in tickers:
        info = run_ticker(predictor, ticker, verbose=args.verbose)
        if info:
            results.append(info)

    if results:
        metrics_df = pd.DataFrame(results)
        metrics_path = PRED_OUT / "_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)

        print(f"\n{'='*60}")
        print(f"  TỔNG KẾT — {len(results)}/{len(tickers)} mã thành công")
        print(f"  Avg MAE   : {metrics_df['mae'].mean():>12,.0f} VND")
        print(f"  Avg MAPE  : {metrics_df['mape'].mean():.2f}%")
        print(f"  Avg DirAcc: {metrics_df['dir_acc'].mean():.1f}%")
        print(f"  Metrics   → {metrics_path}")
        print(f"  Charts    → {CHART_OUT}/")
        print(f"{'='*60}\n")
    else:
        print("\n❌ Không có kết quả nào thành công.")


if __name__ == "__main__":
    main()
