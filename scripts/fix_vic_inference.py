#!/usr/bin/env python3
"""
VIC Outlier Fix — Kronos Inference với Price Normalization
==========================================================
VIC giao dịch ở ~230,000 VND (outlier so với phần lớn VN30 ở 10k–100k VND).
Script này normalize giá về quanh 1.0 trước khi đưa vào Kronos, sau đó
denormalize lại để so sánh với phương pháp gốc.

Output:
  results/predictions/VIC_pred_normalized.csv  — predictions đã denormalize (VND)
  results/charts/VIC_comparison.png            — so sánh Original vs Normalized
"""

import sys
import os
import warnings
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
KRONOS_DIR   = BASE_DIR / "models" / "Kronos-repo"
PROCESSED    = BASE_DIR / "data" / "processed"
PRED_OUT     = BASE_DIR / "results" / "predictions"
CHART_OUT    = BASE_DIR / "results" / "charts"

PRED_OUT.mkdir(parents=True, exist_ok=True)
CHART_OUT.mkdir(parents=True, exist_ok=True)

# Add Kronos source to path
sys.path.insert(0, str(KRONOS_DIR))

# ── Config ────────────────────────────────────────────────────────────────────
TICKER       = "VIC"
LOOKBACK     = 400     # số ngày lịch sử đưa vào context
PRED_LEN     = 20      # số ngày dự báo
SAMPLE_COUNT = 20      # Monte Carlo samples
TEMPERATURE  = 1.0
TOP_P        = 0.9

MODEL_ID     = "NeoQuasar/Kronos-small"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"

OHLC_COLS    = ["open", "high", "low", "close"]


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
    """Tính MAE, MAPE, Direction Accuracy."""
    actual_arr    = actual.values.astype(float)
    predicted_arr = predicted.values.astype(float)

    mae  = float(np.mean(np.abs(actual_arr - predicted_arr)))
    mape = float(np.mean(np.abs((actual_arr - predicted_arr) / actual_arr)) * 100)

    actual_dir = np.sign(np.diff(actual_arr, prepend=actual_arr[0]))
    pred_dir   = np.sign(np.diff(predicted_arr, prepend=predicted_arr[0]))
    dir_acc    = float(np.mean(actual_dir == pred_dir) * 100)

    return {
        "mae":     round(mae, 2),
        "mape":    round(mape, 4),
        "dir_acc": round(dir_acc, 2),
    }


# ── Load and split data ───────────────────────────────────────────────────────
def load_vic_data():
    csv_path = PROCESSED / f"{TICKER}.csv"
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_splits(df: pd.DataFrame):
    """Tính lookback thực tế, trả về context + holdout DataFrames."""
    effective_lookback = min(LOOKBACK, len(df) - PRED_LEN)
    split_idx = len(df) - PRED_LEN

    x_df = df.loc[
        split_idx - effective_lookback : split_idx - 1,
        ["open", "high", "low", "close", "volume", "amount"],
    ].copy().reset_index(drop=True)

    x_timestamp = df.loc[
        split_idx - effective_lookback : split_idx - 1, "date"
    ].reset_index(drop=True)

    holdout_df = df.loc[split_idx : split_idx + PRED_LEN - 1].reset_index(drop=True)

    last_date    = df["date"].iloc[split_idx - 1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=PRED_LEN)
    y_timestamp  = pd.Series(future_dates[:PRED_LEN])

    return x_df, x_timestamp, y_timestamp, holdout_df, future_dates


# ── Run inference (raw, no normalization) ────────────────────────────────────
def run_original(predictor, df: pd.DataFrame):
    """Inference với giá gốc (không normalize) — load từ file đã có."""
    print("\n[1/2] 📊 Loading VIC original prediction (already computed)...")
    pred_path = PRED_OUT / "VIC_pred.csv"
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        print(f"      ✅ Loaded from {pred_path.name}")
        return pred_df
    else:
        # Re-run if not cached
        print("      ⚠️  VIC_pred.csv not found — running original inference...")
        x_df, x_timestamp, y_timestamp, holdout_df, future_dates = get_splits(df)
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=PRED_LEN,
            T=TEMPERATURE,
            top_p=TOP_P,
            sample_count=SAMPLE_COUNT,
            verbose=False,
        )
        pred_df["date"] = future_dates[:PRED_LEN]
        pred_df.to_csv(pred_path, index=False)
        return pred_df


# ── Run inference WITH normalization ─────────────────────────────────────────
def run_normalized(predictor, df: pd.DataFrame):
    """
    Normalize OHLC → [~1.0] bằng cách chia cho giá đóng cửa cuối context,
    chạy Kronos, rồi denormalize về VND thực.
    """
    print("\n[2/2] 🔧 Running VIC normalized inference...")
    x_df, x_timestamp, y_timestamp, holdout_df, future_dates = get_splits(df)

    # Scale factor = giá đóng cửa cuối cùng của context
    scale_factor = float(x_df["close"].iloc[-1])
    print(f"      scale_factor = {scale_factor:,.0f} VND  (close cuối context)")

    # Normalize OHLC columns only; keep volume + amount as-is
    x_df_norm = x_df.copy()
    x_df_norm[OHLC_COLS] = x_df_norm[OHLC_COLS] / scale_factor
    print(f"      OHLC range after norm: "
          f"[{x_df_norm['close'].min():.4f}, {x_df_norm['close'].max():.4f}]")

    pred_df_norm = predictor.predict(
        df=x_df_norm,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=PRED_LEN,
        T=TEMPERATURE,
        top_p=TOP_P,
        sample_count=SAMPLE_COUNT,
        verbose=False,
    )

    # Denormalize OHLC back to VND
    pred_df_denorm = pred_df_norm.copy()
    for col in OHLC_COLS:
        if col in pred_df_denorm.columns:
            pred_df_denorm[col] = pred_df_denorm[col] * scale_factor

    pred_df_denorm["date"] = future_dates[:PRED_LEN]

    # Save
    out_path = PRED_OUT / "VIC_pred_normalized.csv"
    pred_df_denorm.to_csv(out_path, index=False)
    print(f"      ✅ Saved → {out_path.name}")

    return pred_df_denorm, holdout_df


# ── Comparison chart ──────────────────────────────────────────────────────────
def plot_comparison(df: pd.DataFrame,
                    pred_orig: pd.DataFrame,
                    pred_norm: pd.DataFrame,
                    holdout_df: pd.DataFrame,
                    metrics_orig: dict,
                    metrics_norm: dict):
    """2-subplot chart: Original vs Normalized."""
    history_tail = df.iloc[-(PRED_LEN * 2): len(df) - PRED_LEN].copy()  # 40 ngày lịch sử
    history_close = list(history_tail["close"])

    def _make_x(n_hist, n_pred):
        return list(range(n_hist)), list(range(n_hist - 1, n_hist + n_pred))

    n_hist = len(history_tail)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
    fig.suptitle(f"{TICKER} — Kronos Inference: Original vs Price-Normalized",
                 fontsize=15, fontweight="bold", y=1.01)

    titles = ["Original (no normalization)", "With Price Normalization"]
    preds  = [pred_orig, pred_norm]
    mets   = [metrics_orig, metrics_norm]

    for ax, title, pred_df, m in zip(axes, titles, preds, mets):
        # History
        ax.plot(range(n_hist), history_close,
                color="#2196F3", linewidth=1.8, label="Lịch sử (40 ngày)")

        # Prediction (connect from last history point)
        pred_close = [history_close[-1]] + list(pred_df["close"])
        ax.plot(range(n_hist - 1, n_hist + PRED_LEN),
                pred_close,
                color="#F44336", linewidth=2, linestyle="--",
                label=f"Dự báo ({PRED_LEN}D)")

        # Actual holdout
        actual_close = [history_close[-1]] + list(holdout_df["close"])
        ax.plot(range(n_hist - 1, n_hist + len(holdout_df)),
                actual_close,
                color="#4CAF50", linewidth=1.8, label="Thực tế")

        # Vertical separator
        ax.axvline(x=n_hist - 1, color="gray", linestyle=":", linewidth=1.2, alpha=0.6)

        ax.set_title(
            f"{title}\n"
            f"MAE={m['mae']:,.0f} VND  |  MAPE={m['mape']:.2f}%  |  DirAcc={m['dir_acc']:.1f}%",
            fontsize=11,
        )
        ax.set_xlabel("Ngày (tương đối)")
        ax.set_ylabel("Giá đóng cửa (VND)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = CHART_OUT / "VIC_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📈 Chart saved → {out_path}")
    return out_path


# ── Print comparison table ────────────────────────────────────────────────────
def print_comparison(metrics_orig: dict, metrics_norm: dict):
    print("\n" + "=" * 60)
    print(f"  VIC — Original vs Normalized Inference")
    print("=" * 60)
    print(f"  {'Metric':<22} {'Original':>14} {'Normalized':>14}  {'Δ':>10}")
    print("  " + "-" * 56)

    mae_delta  = metrics_norm["mae"]  - metrics_orig["mae"]
    mape_delta = metrics_norm["mape"] - metrics_orig["mape"]
    dir_delta  = metrics_norm["dir_acc"] - metrics_orig["dir_acc"]

    print(f"  {'MAE (VND)':<22} {metrics_orig['mae']:>14,.0f} {metrics_norm['mae']:>14,.0f}  {mae_delta:>+10,.0f}")
    print(f"  {'MAPE (%)':<22} {metrics_orig['mape']:>14.4f} {metrics_norm['mape']:>14.4f}  {mape_delta:>+10.4f}")
    print(f"  {'Direction Acc (%)':<22} {metrics_orig['dir_acc']:>14.1f} {metrics_norm['dir_acc']:>14.1f}  {dir_delta:>+10.1f}")
    print("=" * 60)

    # Conclusion
    print("\n📌 Kết luận:")
    if metrics_norm["mape"] < metrics_orig["mape"]:
        improvement = (metrics_orig["mape"] - metrics_norm["mape"]) / metrics_orig["mape"] * 100
        print(f"   ✅ Normalization CẢI THIỆN MAPE {improvement:.1f}%: "
              f"{metrics_orig['mape']:.2f}% → {metrics_norm['mape']:.2f}%")
        print(f"   ✅ MAE: {metrics_orig['mae']:,.0f} → {metrics_norm['mae']:,.0f} VND")
        if metrics_norm["dir_acc"] >= metrics_orig["dir_acc"]:
            print(f"   ✅ Direction Accuracy cũng tốt hơn hoặc tương đương.")
        print(f"\n   💡 Nên dùng price normalization cho VIC (và các mã giá cao khác).")
    else:
        degradation = (metrics_norm["mape"] - metrics_orig["mape"]) / metrics_orig["mape"] * 100
        print(f"   ⚠️  Normalization KHÔNG cải thiện MAPE (+{degradation:.1f}%): "
              f"{metrics_orig['mape']:.2f}% → {metrics_norm['mape']:.2f}%")
        print(f"\n   💡 VIC có thể cần chiến lược khác (log-transform, diff, fine-tune).")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("\n" + "=" * 60)
    print(f"  VIC Outlier Fix — Price Normalization Experiment")
    print(f"  Lookback={LOOKBACK}D  Pred={PRED_LEN}D  Samples={SAMPLE_COUNT}")
    print(f"  Device: {device}")
    print("=" * 60)

    # 1. Load data
    df = load_vic_data()
    print(f"\n📂 VIC data: {len(df)} rows  "
          f"({df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()})")
    print(f"   Close range: {df['close'].min():,.0f} – {df['close'].max():,.0f} VND")

    # 2. Load Kronos
    predictor = load_kronos(device=device)

    # 3. Get splits (for holdout — shared between both runs)
    _, _, _, holdout_df, _ = get_splits(df)

    # 4. Original inference (load cached or re-run)
    pred_orig = run_original(predictor, df)
    metrics_orig = calc_metrics(holdout_df["close"], pred_orig["close"])
    print(f"      Original  → MAE={metrics_orig['mae']:,.0f}  "
          f"MAPE={metrics_orig['mape']:.2f}%  DirAcc={metrics_orig['dir_acc']:.1f}%")

    # 5. Normalized inference
    pred_norm, holdout_df = run_normalized(predictor, df)
    metrics_norm = calc_metrics(holdout_df["close"], pred_norm["close"])
    print(f"      Normalized→ MAE={metrics_norm['mae']:,.0f}  "
          f"MAPE={metrics_norm['mape']:.2f}%  DirAcc={metrics_norm['dir_acc']:.1f}%")

    # 6. Plot comparison
    plot_comparison(df, pred_orig, pred_norm, holdout_df, metrics_orig, metrics_norm)

    # 7. Print comparison table + conclusion
    print_comparison(metrics_orig, metrics_norm)


if __name__ == "__main__":
    main()
