"""
Script làm sạch và chuẩn hóa dữ liệu OHLCV cho Kronos.
Chạy: python scripts/preprocess.py
"""

import os
import csv
import json
from datetime import datetime

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

VN30 = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]


def load_csv(symbol: str) -> list[dict]:
    filepath = os.path.join(RAW_DIR, f"{symbol}.csv")
    with open(filepath, newline="") as f:
        return list(csv.DictReader(f))


def clean(rows: list[dict]) -> list[dict]:
    """
    - Dùng adj_close thay close (đã điều chỉnh corporate actions)
    - Tính amount nếu chưa có
    - Loại bỏ ngày volume = 0
    - Sắp xếp theo ngày tăng dần
    """
    cleaned = []
    for r in rows:
        try:
            volume = int(r["volume"])
            if volume == 0:
                continue
            close = float(r["adj_close"] or r["close"])
            open_ = float(r["open"])
            high  = float(r["high"])
            low   = float(r["low"])
            amount = float(r.get("amount") or volume * close)

            cleaned.append({
                "date":   r["date"],
                "open":   round(open_, 2),
                "high":   round(high, 2),
                "low":    round(low, 2),
                "close":  round(close, 2),
                "volume": volume,
                "amount": round(amount, 0),
            })
        except (ValueError, KeyError):
            continue

    cleaned.sort(key=lambda x: x["date"])
    return cleaned


def save_processed(symbol: str, rows: list[dict]):
    os.makedirs(OUT_DIR, exist_ok=True)
    filepath = os.path.join(OUT_DIR, f"{symbol}.csv")
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "open", "high", "low", "close", "volume", "amount"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("🧹 Tiền xử lý dữ liệu VN30\n")
    stats = {}

    for symbol in VN30:
        try:
            raw   = load_csv(symbol)
            clean_rows = clean(raw)
            save_processed(symbol, clean_rows)
            stats[symbol] = {
                "raw": len(raw),
                "clean": len(clean_rows),
                "dropped": len(raw) - len(clean_rows),
                "from": clean_rows[0]["date"] if clean_rows else "?",
                "to":   clean_rows[-1]["date"] if clean_rows else "?",
            }
            print(f"  ✅ {symbol:4s}: {len(clean_rows):,} ngày ({stats[symbol]['from']} → {stats[symbol]['to']}) | loại bỏ {stats[symbol]['dropped']} rows")
        except Exception as e:
            print(f"  ❌ {symbol}: {e}")

    # Lưu stats
    stats_path = os.path.join(OUT_DIR, "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    total = sum(v["clean"] for v in stats.values())
    print(f"\n{'='*40}")
    print(f"📊 Tổng records sạch : {total:,}")
    print(f"📁 Lưu tại           : {os.path.abspath(OUT_DIR)}")
    print(f"📋 Stats             : _stats.json")


if __name__ == "__main__":
    main()

