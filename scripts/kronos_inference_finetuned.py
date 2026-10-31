#!/usr/bin/env python3
"""
Kronos Fine-tuned Inference + Zero-Shot vs Fine-tuned Comparison
================================================================
Load checkpoint từ local path (thay vì HuggingFace) và chạy inference
trên toàn bộ VN30, sau đó so sánh với zero-shot metrics.

Usage:
  python scripts/kronos_inference_finetuned.py
  python scripts/kronos_inference_finetuned.py --tickers VCB HPG
"""

import sys, os, argparse, warnings
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
KRONOS_DIR  = BASE_DIR / "models" / "Kronos-repo"
PROCESSED   = BASE_DIR / "data" / "processed"
ZS_PRED     = BASE_DIR / "results" / "predictions"          # zero-shot
FT_PRED     = BASE_DIR / "results" / "predictions_finetuned"
FT_CHARTS   = BASE_DIR / "results" / "charts_finetuned"
CMP_OUT     = BASE_DIR / "results" / "analysis"

CKPT_BASE   = BASE_DIR / "models" / "finetuned" / "vn30_full_3ep"
TOK_CKPT    = CKPT_BASE / "tokenizer" / "best_model"
PRED_CKPT   = CKPT_BASE / "basemodel" / "best_model"

# Note: overridden by --exp-name / --model-dir / --tok-dir args in main()

for d in [FT_PRED, FT_CHARTS, CMP_OUT]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(KRONOS_DIR))

# ── Config ────────────────────────────────────────────────────────────────────
LOOKBACK     = 400
PRED_LEN     = 20
SAMPLE_COUNT = 20
TEMPERATURE  = 1.0
TOP_P        = 0.9

ALL_TICKERS = [
    "ACB","BCM","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG",
    "MBB","MSN","MWG","PLX","POW","SAB","SHB","SSB","SSI","STB",
    "TCB","TPB","VCB","VHM","VIB","VIC","VJC","VNM","VPB","VRE",
]


# ── Load fine-tuned Kronos ────────────────────────────────────────────────────
def load_finetuned(device: str = "cpu"):
    from model import Kronos, KronosTokenizer, KronosPredictor

    print(f"📥 Loading fine-tuned tokenizer from {TOK_CKPT} ...")
    tokenizer = KronosTokenizer.from_pretrained(str(TOK_CKPT))

    print(f"📥 Loading fine-tuned predictor from {PRED_CKPT} ...")
    model = Kronos.from_pretrained(str(PRED_CKPT))

    print(f"🔧 Device: {device}")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    return predictor


# ── Metrics ───────────────────────────────────────────────────────────────────
def calc_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    a, p = actual.values, predicted.values
    mae  = float(np.mean(np.abs(a - p)))
    mape = float(np.mean(np.abs((a - p) / a)) * 100)
    ad   = np.sign(np.diff(a, prepend=a[0]))
    pd_  = np.sign(np.diff(p, prepend=p[0]))
    dir_acc = float(np.mean(ad == pd_) * 100)
    return {"mae": round(mae, 2), "mape": round(mape, 4), "dir_acc": round(dir_acc, 2)}


# ── Run inference per ticker ──────────────────────────────────────────────────
def run_ticker(predictor, ticker: str) -> dict | None:
    csv_path = PROCESSED / f"{ticker}.csv"
    if not csv_path.exists():
        print(f"  ❌ {ticker}: file not found"); return None

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < LOOKBACK + PRED_LEN:
        print(f"  ⚠️  {ticker}: not enough data"); return None

    split_idx   = len(df) - PRED_LEN
    x_df        = df.loc[split_idx - LOOKBACK : split_idx - 1,
                         ["open","high","low","close","volume","amount"]].copy().reset_index(drop=True)
    x_timestamp = df.loc[split_idx - LOOKBACK : split_idx - 1, "date"].reset_index(drop=True)
    last_date   = df["date"].iloc[split_idx - 1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=PRED_LEN)
    y_timestamp  = pd.Series(future_dates[:PRED_LEN])
    actual_df    = df.loc[split_idx : split_idx + PRED_LEN - 1].reset_index(drop=True)

    try:
        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=PRED_LEN, T=TEMPERATURE, top_p=TOP_P,
            sample_count=SAMPLE_COUNT, verbose=False,
        )
        pred_df["date"] = future_dates[:PRED_LEN]
        pred_df.to_csv(FT_PRED / f"{ticker}_ft_pred.csv", index=False)

        metrics = calc_metrics(actual_df["close"], pred_df["close"])
        print(f"  ✅ {ticker:5s}  MAE={metrics['mae']:>10,.0f}  "
              f"MAPE={metrics['mape']:5.2f}%  DirAcc={metrics['dir_acc']:5.1f}%")
        return {"ticker": ticker, **metrics}

    except Exception as e:
        print(f"  ❌ {ticker}: {e}"); return None


# ── Comparison plot & report ──────────────────────────────────────────────────
def compare_and_report(zs_df: pd.DataFrame, ft_df: pd.DataFrame):
    merged = zs_df.merge(ft_df, on="ticker", suffixes=("_zs", "_ft"))

    # ── 1. Bar chart: MAPE comparison ────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Zero-Shot vs Fine-tuned — Kronos VN30", fontsize=14, fontweight="bold")

    x = np.arange(len(merged))
    w = 0.35
    metrics_info = [
        ("mape",    "MAPE (%)",             "lower=better", "%.2f%%"),
        ("dir_acc", "Direction Acc (%)",    "higher=better", "%.1f%%"),
        ("mae",     "MAE (VND)",            "lower=better",  "%,.0f"),
    ]

    for ax, (col, label, note, fmt) in zip(axes, metrics_info):
        zs_vals = merged[f"{col}_zs"].values
        ft_vals = merged[f"{col}_ft"].values
        bars_zs = ax.bar(x - w/2, zs_vals, w, label="Zero-Shot", color="#90CAF9", alpha=0.85)
        bars_ft = ax.bar(x + w/2, ft_vals, w, label="Fine-tuned", color="#EF9A9A", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(merged["ticker"], rotation=90, fontsize=7)
        ax.set_title(f"{label}\n({note})", fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
        if col == "mae":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v/1000:.0f}k"))

    plt.tight_layout()
    out = CMP_OUT / "zs_vs_ft_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n📊 Comparison chart → {out}")

    # ── 2. Delta table ────────────────────────────────────────────────────────
    merged["mape_delta"]    = merged["mape_ft"]    - merged["mape_zs"]     # negative = better
    merged["dir_acc_delta"] = merged["dir_acc_ft"] - merged["dir_acc_zs"]  # positive = better
    merged["mae_delta"]     = merged["mae_ft"]     - merged["mae_zs"]      # negative = better

    # ── 3. Summary stats ──────────────────────────────────────────────────────
    improved_mape   = (merged["mape_delta"]    < 0).sum()
    improved_dir    = (merged["dir_acc_delta"] > 0).sum()
    avg_mape_zs     = merged["mape_zs"].mean()
    avg_mape_ft     = merged["mape_ft"].mean()
    avg_dir_zs      = merged["dir_acc_zs"].mean()
    avg_dir_ft      = merged["dir_acc_ft"].mean()

    # ── 4. Markdown report ────────────────────────────────────────────────────
    report_lines = [
        "# Kronos VN30 — Zero-Shot vs Fine-tuned Comparison",
        f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## 📊 Tổng quan",
        "",
        "| Metric | Zero-Shot | Fine-tuned | Delta |",
        "|--------|-----------|------------|-------|",
        f"| Avg MAPE     | {avg_mape_zs:.2f}%  | {avg_mape_ft:.2f}%  | {avg_mape_ft-avg_mape_zs:+.2f}% |",
        f"| Avg DirAcc   | {avg_dir_zs:.1f}%   | {avg_dir_ft:.1f}%   | {avg_dir_ft-avg_dir_zs:+.1f}% |",
        f"| Mã MAPE tốt hơn | — | {improved_mape}/30 | |",
        f"| Mã DirAcc tốt hơn | — | {improved_dir}/30 | |",
        "",
        "## 🏆 Top 10 cải thiện nhiều nhất (DirAcc)",
        "",
        "| Ticker | ZS DirAcc | FT DirAcc | Delta | ZS MAPE | FT MAPE |",
        "|--------|-----------|-----------|-------|---------|---------|",
    ]
    top10 = merged.nlargest(10, "dir_acc_delta")
    for _, r in top10.iterrows():
        sign = "✅" if r["dir_acc_delta"] > 0 else "❌"
        report_lines.append(
            f"| {r['ticker']} | {r['dir_acc_zs']:.0f}% | {r['dir_acc_ft']:.0f}% | "
            f"{sign} {r['dir_acc_delta']:+.0f}% | {r['mape_zs']:.2f}% | {r['mape_ft']:.2f}% |"
        )

    report_lines += [
        "",
        "## ⚠️ Mã bị regression (FT tệ hơn ZS về DirAcc)",
        "",
        "| Ticker | ZS DirAcc | FT DirAcc | Delta |",
        "|--------|-----------|-----------|-------|",
    ]
    regressed = merged[merged["dir_acc_delta"] < 0].sort_values("dir_acc_delta")
    for _, r in regressed.iterrows():
        report_lines.append(
            f"| {r['ticker']} | {r['dir_acc_zs']:.0f}% | {r['dir_acc_ft']:.0f}% | {r['dir_acc_delta']:+.0f}% |"
        )

    report_lines += [
        "",
        "## 📋 Full Comparison Table",
        "",
        "| Ticker | ZS MAPE | FT MAPE | ΔMAPE | ZS DirAcc | FT DirAcc | ΔDirAcc |",
        "|--------|---------|---------|-------|-----------|-----------|---------|",
    ]
    for _, r in merged.sort_values("dir_acc_delta", ascending=False).iterrows():
        m_sign = "✅" if r["mape_delta"] < 0 else "❌"
        d_sign = "✅" if r["dir_acc_delta"] > 0 else "❌"
        report_lines.append(
            f"| {r['ticker']} | {r['mape_zs']:.2f}% | {r['mape_ft']:.2f}% | "
            f"{m_sign}{r['mape_delta']:+.2f}% | {r['dir_acc_zs']:.0f}% | "
            f"{r['dir_acc_ft']:.0f}% | {d_sign}{r['dir_acc_delta']:+.0f}% |"
        )

    report_path = CMP_OUT / "zs_vs_ft_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"📝 Report → {report_path}")

    # Print summary to console
    print(f"""
╔══════════════════════════════════════════════════════════╗
║        ZERO-SHOT  vs  FINE-TUNED  —  VN30 SUMMARY        ║
╠══════════════════════════════════════════════════════════╣
║  Avg MAPE    : {avg_mape_zs:5.2f}%  →  {avg_mape_ft:5.2f}%   ({avg_mape_ft-avg_mape_zs:+.2f}%)       ║
║  Avg DirAcc  : {avg_dir_zs:5.1f}%  →  {avg_dir_ft:5.1f}%   ({avg_dir_ft-avg_dir_zs:+.1f}%)         ║
║  Mã MAPE↓   : {improved_mape:2d}/30                                    ║
║  Mã DirAcc↑  : {improved_dir:2d}/30                                    ║
╚══════════════════════════════════════════════════════════╝
""")
    return merged


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global SAMPLE_COUNT

    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--samples", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Fine-tune experiment name under models/finetuned/")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Path to predictor best_model dir (overrides --exp-name)")
    parser.add_argument("--tok-dir", type=str, default=None,
                        help="Path to tokenizer best_model dir (overrides --exp-name)")
    args = parser.parse_args()

    SAMPLE_COUNT = args.samples
    tickers = args.tickers or ALL_TICKERS

    # Override checkpoint paths if requested
    global TOK_CKPT, PRED_CKPT
    if args.exp_name:
        exp_base = BASE_DIR / "models" / "finetuned" / args.exp_name
        TOK_CKPT  = exp_base / "tokenizer" / "best_model"
        PRED_CKPT = exp_base / "basemodel"  / "best_model"
    if args.model_dir:
        PRED_CKPT = Path(args.model_dir)
    if args.tok_dir:
        TOK_CKPT  = Path(args.tok_dir)

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  Kronos VN30 — Fine-tuned Inference")
    print(f"  Checkpoint: {PRED_CKPT}")
    print(f"  Mã: {', '.join(tickers)}")
    print(f"  Device: {device}  Samples: {SAMPLE_COUNT}")
    print(f"{'='*60}\n")

    predictor = load_finetuned(device=device)
    print()

    ft_results = []
    for ticker in tickers:
        info = run_ticker(predictor, ticker)
        if info:
            ft_results.append(info)

    if not ft_results:
        print("❌ Không có kết quả."); return

    ft_df = pd.DataFrame(ft_results)
    ft_df.to_csv(FT_PRED / "_metrics_finetuned.csv", index=False)

    print(f"\n✅ Fine-tuned inference: {len(ft_results)}/{len(tickers)} mã xong")
    print(f"   Avg MAPE: {ft_df['mape'].mean():.2f}%  |  Avg DirAcc: {ft_df['dir_acc'].mean():.1f}%")

    # Compare với zero-shot nếu có
    zs_metrics_path = ZS_PRED / "_metrics.csv"
    if zs_metrics_path.exists():
        print("\n🔄 So sánh với zero-shot...\n")
        zs_df = pd.read_csv(zs_metrics_path)
        zs_df = zs_df[zs_df["ticker"].isin(tickers)]
        if not zs_df.empty:
            compare_and_report(zs_df, ft_df)
    else:
        print("⚠️  Zero-shot metrics không tìm thấy — bỏ qua so sánh.")


if __name__ == "__main__":
    main()
