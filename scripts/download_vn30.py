"""
Script thu thập dữ liệu OHLCV cho VN30 từ Yahoo Finance.
Chạy: python scripts/download_vn30.py
"""

import urllib.request
import json
import ssl
import time
import csv
import os
from datetime import datetime

# ─── Config ───────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RANGE   = "5y"   # 5 năm
INTERVAL = "1d"  # Daily

VN30 = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]
# ──────────────────────────────────────────────────────────


def fetch_ohlcv(symbol: str) -> list[dict]:
    """Lấy OHLCV từ Yahoo Finance, trả về list of dict."""
    ticker = f"{symbol}.VN"
    url = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval={INTERVAL}&range={RANGE}"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
        data = json.loads(resp.read().decode())

    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    q = result["indicators"]["quote"][0]
    adj_list = result["indicators"].get("adjclose", [{}])[0].get("adjclose", [])

    rows = []
    for i, t in enumerate(timestamps):
        o, h, l, c, v = q["open"][i], q["high"][i], q["low"][i], q["close"][i], q["volume"][i]
        if None in (o, h, l, c, v):
            continue
        a = adj_list[i] if adj_list and i < len(adj_list) else None
        rows.append({
            "date":      datetime.fromtimestamp(t).strftime("%Y-%m-%d"),
            "open":      round(o, 2),
            "high":      round(h, 2),
            "low":       round(l, 2),
            "close":     round(c, 2),
            "volume":    int(v),
            "adj_close": round(a, 2) if a else "",
            # amount = volume × close (VND) — cần cho Kronos
            "amount":    round(int(v) * c, 0),
        })
    return rows


def save_csv(symbol: str, rows: list[dict]) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    filepath = os.path.join(OUT_DIR, f"{symbol}.csv")
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "open", "high", "low", "close", "volume", "adj_close", "amount"])
        writer.writeheader()
        writer.writerows(rows)
    return filepath


def main():
    print(f"📥 Thu thập dữ liệu VN30 ({RANGE}, {INTERVAL}) từ Yahoo Finance\n")
    success, failed = [], []

    for symbol in VN30:
        try:
            rows = fetch_ohlcv(symbol)
            filepath = save_csv(symbol, rows)
            print(f"  ✅ {symbol:4s}: {len(rows):,} ngày → {os.path.basename(filepath)}")
            success.append(symbol)
        except Exception as e:
            print(f"  ❌ {symbol:4s}: {e}")
            failed.append(symbol)
        time.sleep(0.7)  # tránh rate limit

    print(f"\n{'='*40}")
    print(f"✅ Thành công : {len(success)}/{len(VN30)} mã")
    if failed:
        print(f"❌ Thất bại   : {failed}")
    total = sum(
        sum(1 for _ in open(os.path.join(OUT_DIR, f"{s}.csv"))) - 1
        for s in success
    )
    print(f"📊 Tổng records: {total:,}")
    print(f"📁 Lưu tại     : {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()

