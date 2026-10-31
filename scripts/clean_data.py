#!/usr/bin/env python3
"""
VN30 OHLCV Data Cleaning & Normalization Pipeline
==================================================
Input:  data/raw/*.csv        — raw Yahoo Finance data
Output: data/processed/*.csv  — cleaned, normalized, Kronos-ready

Steps:
  1. Parse dates → datetime, sort ascending
  2. Remove zero-volume rows (non-trading days / data artifacts)
  3. Adjust all OHLC using adj_close ratio (corporate-action adjusted prices)
  4. Compute `amount` = volume × adj_close  (VND, raw)
  5. Round prices to 2 dp, volume & amount to 0 dp
  6. Validate OHLC logic (high >= max(open,close), low <= min(open,close))
  7. Fill any remaining NaNs with forward-fill then backward-fill
  8. Save to data/processed/<TICKER>.csv
  9. Print summary report
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
RAW_DIR    = BASE_DIR / "data" / "raw"
OUT_DIR    = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
PRICE_SCALE = 1000          # Yahoo trả về giá ÷ 1000 so với thực tế HOSE? No —
                             # Yahoo đã trả VND thực; set =1 để không scale
                             # Đổi thành 1000 nếu muốn chuyển về đơn vị nghìn đồng

# ── Helpers ──────────────────────────────────────────────────────────────────

def adjust_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Điều chỉnh OHLC theo tỷ lệ adj_close/close (split/dividend adjustment).
    Giữ nguyên volume và adj_close, drop cột close gốc.
    """
    ratio = df["adj_close"] / df["close"]
    for col in ["open", "high", "low"]:
        df[col] = (df[col] * ratio).round(2)
    df["close"] = df["adj_close"].round(2)
    df = df.drop(columns=["adj_close"])
    return df


def compute_amount(df: pd.DataFrame) -> pd.DataFrame:
    """amount = volume × close (VND, raw)"""
    df["amount"] = (df["volume"] * df["close"]).round(0).astype(np.int64)
    return df


def validate_ohlc(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Phát hiện và sửa các vi phạm OHLC logic:
      - high < max(open, close)  → đặt high = max(open, close)
      - low  > min(open, close)  → đặt low  = min(open, close)
    """
    bad_high = df["high"] < df[["open", "close"]].max(axis=1)
    bad_low  = df["low"]  > df[["open", "close"]].min(axis=1)

    n_bad = bad_high.sum() + bad_low.sum()
    if n_bad > 0:
        print(f"  ⚠️  {ticker}: fixing {bad_high.sum()} bad highs, {bad_low.sum()} bad lows")
        df.loc[bad_high, "high"] = df.loc[bad_high, ["open", "close"]].max(axis=1)
        df.loc[bad_low,  "low"]  = df.loc[bad_low,  ["open", "close"]].min(axis=1)

    return df


def clean_ticker(path: str) -> dict:
    ticker = Path(path).stem
    df = pd.read_csv(path)

    # 1. Parse dates
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 2. Remove zero-volume rows (non-trading days / holidays)
    n_zero_vol = (df["volume"] == 0).sum()
    df = df[df["volume"] > 0].copy()

    # 3. Adjust OHLC (corporate-action adjusted)
    df = adjust_ohlc(df)

    # 4. Validate OHLC logic
    df = validate_ohlc(df, ticker)

    # 5. Fill NaN (forward then backward)
    n_nan = df.isnull().sum().sum()
    if n_nan > 0:
        df = df.ffill().bfill()

    # 6. Compute amount
    df = compute_amount(df)

    # 7. Reorder columns → Kronos-friendly format
    df = df[["date", "open", "high", "low", "close", "volume", "amount"]]

    # 8. Save
    out_path = OUT_DIR / f"{ticker}.csv"
    df.to_csv(out_path, index=False, date_format="%Y-%m-%d")

    return {
        "ticker":     ticker,
        "rows":       len(df),
        "zero_vol_dropped": n_zero_vol,
        "nan_fixed":  n_nan,
        "date_start": df["date"].min().date(),
        "date_end":   df["date"].max().date(),
        "price_min":  df["close"].min(),
        "price_max":  df["close"].max(),
        "vol_avg":    int(df["volume"].mean()),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    files = sorted(glob.glob(str(RAW_DIR / "*.csv")))
    if not files:
        print(f"❌ Không tìm thấy file CSV trong {RAW_DIR}")
        return

    print(f"🔄 Đang làm sạch {len(files)} mã VN30...\n")

    results = []
    for f in files:
        try:
            info = clean_ticker(f)
            results.append(info)
            print(f"  ✅ {info['ticker']:5s}  {info['rows']:4d} rows  "
                  f"{info['date_start']} → {info['date_end']}  "
                  f"zero-vol dropped={info['zero_vol_dropped']}")
        except Exception as e:
            print(f"  ❌ {Path(f).stem}: {e}")

    # Summary
    total_rows = sum(r["rows"] for r in results)
    total_dropped = sum(r["zero_vol_dropped"] for r in results)

    print(f"""
╔══════════════════════════════════════════════════════╗
║           VN30 DATA CLEANING — SUMMARY               ║
╠══════════════════════════════════════════════════════╣
║  Mã xử lý thành công : {len(results):2d} / {len(files)}                    ║
║  Tổng rows giữ lại   : {total_rows:,}                   ║
║  Zero-vol rows dropped: {total_dropped:,}                    ║
║  Output              : data/processed/              ║
╚══════════════════════════════════════════════════════╝

Columns trong file processed:
  date       — YYYY-MM-DD (ngày giao dịch)
  open       — Giá mở cửa đã điều chỉnh (VND)
  high       — Giá cao nhất đã điều chỉnh (VND)
  low        — Giá thấp nhất đã điều chỉnh (VND)
  close      — Giá đóng cửa đã điều chỉnh (VND)  ← dùng cột này cho model
  volume     — Khối lượng giao dịch (cổ phiếu)
  amount     — Giá trị giao dịch (VND) = volume × close
""")

    # Save summary CSV
    summary_df = pd.DataFrame(results)
    summary_path = OUT_DIR / "_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"📊 Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
