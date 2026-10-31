#!/usr/bin/env python3
"""
Prepare Fine-tune Data — VN30
================================
Đọc các file CSV trong data/processed/ và tạo sliding-window snapshots
lưu dưới dạng .npz vào data/finetune/train/ và data/finetune/val/.

Mục đích: caching dữ liệu để fine-tuning nhanh hơn (tuỳ chọn).
Thực tế finetune_kronos.py có thể dùng CSV trực tiếp (không cần script này).

Usage:
  python scripts/prepare_finetune_data.py
  python scripts/prepare_finetune_data.py --tickers VCB HPG --lookback 400
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data" / "processed"
FT_TRAIN   = BASE_DIR / "data" / "finetune" / "train"
FT_VAL     = BASE_DIR / "data" / "finetune" / "val"

FEATURE_COLS = ["open", "high", "low", "close", "volume", "amount"]
TRAIN_END    = "2025-10-31"
VAL_START    = "2025-11-01"

# Timestamp format expected by Kronos TemporalEmbedding:
# [minute(0-59), hour(0-23), weekday(0-6), day(1-31), month(1-12)]
STAMP_COLS = ["te_minute", "te_hour", "te_weekday", "te_day", "te_month"]


def make_windows(
    df: pd.DataFrame,
    split: str,
    lookback: int,
    pred_len: int,
    stride: int,
    clip: float,
):
    feat  = df[FEATURE_COLS].values.astype(np.float32)
    stamp = df[STAMP_COLS].values.astype(np.int64)
    n     = len(df)
    dates = pd.to_datetime(df["date"])

    train_end_pos = int((dates <= TRAIN_END).sum())

    if split == "train":
        window      = lookback + pred_len + 1
        start_range = range(0, train_end_pos - window + 1, stride)
    else:
        val_rows        = n - train_end_pos
        eff_lookback    = min(lookback, max(pred_len * 3, val_rows // 2))
        window          = eff_lookback + pred_len + 1
        first_start     = max(0, train_end_pos - eff_lookback)
        eff_stride      = max(1, stride // 2)
        start_range     = range(first_start, n - window + 1, eff_stride)

    if n < window:
        return [], []

    xs, stamps = [], []
    for start in start_range:
        x_seg = feat[start : start + window]
        s_seg = stamp[start : start + window]
        # per-window z-score normalisation
        mu    = x_seg.mean(axis=0)
        sigma = x_seg.std(axis=0)
        x_norm = np.clip((x_seg - mu) / (sigma + 1e-5), -clip, clip)
        xs.append(x_norm)
        stamps.append(s_seg)

    return xs, stamps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers",  nargs="+", default=None)
    parser.add_argument("--lookback", type=int, default=400)
    parser.add_argument("--pred-len", type=int, default=20)
    parser.add_argument("--stride",   type=int, default=5)
    parser.add_argument("--clip",     type=float, default=5.0)
    args = parser.parse_args()

    FT_TRAIN.mkdir(parents=True, exist_ok=True)
    FT_VAL.mkdir(parents=True, exist_ok=True)

    all_tickers = sorted(f.stem for f in DATA_DIR.glob("*.csv") if f.stem != "_summary")
    tickers = [t.upper() for t in (args.tickers or all_tickers)]

    total_train, total_val = 0, 0

    for ticker in tickers:
        csv_path = DATA_DIR / f"{ticker}.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping")
            continue

        df = pd.read_csv(csv_path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill()

        # Add stamp columns
        df["te_minute"]  = np.int64(0)
        df["te_hour"]    = np.int64(9)
        df["te_weekday"] = df["date"].dt.weekday.astype(np.int64)
        df["te_day"]     = df["date"].dt.day.astype(np.int64)
        df["te_month"]   = df["date"].dt.month.astype(np.int64)

        for split, out_dir in [("train", FT_TRAIN), ("val", FT_VAL)]:
            xs, stamps = make_windows(df, split, args.lookback, args.pred_len,
                                      args.stride, args.clip)
            if not xs:
                print(f"  {ticker} [{split}]: 0 windows (skipped)")
                continue

            out_path = out_dir / f"{ticker}.npz"
            np.savez_compressed(
                str(out_path),
                x=np.stack(xs),        # (N, window, 6)
                stamp=np.stack(stamps), # (N, window, 5)
            )
            n_win = len(xs)
            if split == "train":
                total_train += n_win
            else:
                total_val   += n_win
            print(f"  {ticker} [{split}]: {n_win} windows → {out_path}")

    print(f"\n✓ Done!  Train: {total_train} windows  Val: {total_val} windows")
    print(f"  Train dir: {FT_TRAIN}")
    print(f"  Val   dir: {FT_VAL}")


if __name__ == "__main__":
    main()
