"""
Rolling zero-shot Kronos evaluation over many non-overlapping windows.

For each ticker, walks back N_WINDOWS x PRED_LEN days from the end. Window k
predicts df[s:s+PRED_LEN] from context df[s-LOOKBACK:s]. Predicted and actual
close paths are saved so baselines and backtests can reuse them.
Resumable: finished (ticker, window) pairs are skipped.

Usage: python scripts/rolling_eval.py [--windows 24] [--tickers VCB,FPT] [--samples 10]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "models" / "Kronos-repo"))
from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

PROC = BASE / "data" / "processed"
PRED_LEN, SEED = 20, 0
COLS = ["open", "high", "low", "close", "volume", "amount"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=24)
    ap.add_argument("--tickers", default="")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--ft", default="", help="fine-tune exp name under models/finetuned (default: zero-shot)")
    ap.add_argument("--min-start", default="", help="skip windows starting before this date (use 2025-11-01 for FT: train ends 2025-10-31)")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--lookback", type=int, default=400)
    ap.add_argument("--logret", action="store_true", help="feed log-returns, reconstruct prices (kronos_logreturn_inference transforms)")
    ap.add_argument("--tag", default="", help="output suffix paths_<tag>.csv (default: paths.csv, or paths_ft_<exp> with --ft)")
    ap.add_argument("--real-dates", action="store_true", help="y_timestamp = actual trading dates of the target (default: bdate_range)")
    a = ap.parse_args()
    LOOKBACK = a.lookback
    if a.logret:
        sys.path.insert(0, str(BASE / "scripts"))
        import kronos_logreturn_inference as lr

    tickers = a.tickers.split(",") if a.tickers else sorted(p.stem for p in PROC.glob("*.csv") if not p.stem.startswith("_"))
    out_path = BASE / "results" / "rolling" / (f"paths_{a.tag}.csv" if a.tag else f"paths_ft_{a.ft}.csv" if a.ft else "paths.csv")
    ck = BASE / "models" / "finetuned" / a.ft
    model_src = str(ck / "basemodel" / "best_model") if a.ft else "NeoQuasar/Kronos-small"
    tok_src = str(ck / "tokenizer" / "best_model") if a.ft else "NeoQuasar/Kronos-Tokenizer-base"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pred = KronosPredictor(Kronos.from_pretrained(model_src), KronosTokenizer.from_pretrained(tok_src),
                           device=device, max_context=512)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        d = pd.read_csv(out_path, usecols=["ticker", "window"])
        done = set(zip(d.ticker, d.window))

    for t in tickers:
        df = pd.read_csv(PROC / f"{t}.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        n, rows = len(df), []
        for k in range(a.windows, 0, -1):  # window id = k, oldest first
            s = n - k * PRED_LEN
            if s < LOOKBACK or (t, k) in done or (a.min_start and str(df.date.iloc[s].date()) < a.min_start):
                continue
            if s < LOOKBACK + 1:
                continue
            ctx, tgt = df.iloc[s - LOOKBACK:s], df.iloc[s:s + PRED_LEN]
            if len(tgt) < PRED_LEN:
                continue
            torch.manual_seed(SEED + k)
            # ponytail: bdate_range ignores VN holidays; Kronos only uses the calendar features
            fut = tgt.date.reset_index(drop=True) if a.real_dates else pd.Series(pd.bdate_range(pd.Timestamp(ctx.date.iloc[-1]) + pd.Timedelta(days=1), periods=PRED_LEN))
            kw = dict(y_timestamp=fut, pred_len=PRED_LEN, T=a.temp, top_p=0.9, sample_count=a.samples, verbose=False)
            if a.logret:
                raw = df.iloc[s - LOOKBACK - 1:s][COLS].reset_index(drop=True)
                lr_df, last_close, min_pos = lr.to_log_returns(raw)
                lr_df = lr_df.tail(LOOKBACK).reset_index(drop=True)
                lp = pred.predict(df=lr_df, x_timestamp=ctx.date.reset_index(drop=True), **kw)
                p = lr.from_log_returns(lp, last_close, min_pos, vol_ref=max(raw.volume.median(), 1.0))
            else:
                p = pred.predict(df=ctx[COLS], x_timestamp=ctx.date.reset_index(drop=True), **kw)
            rows.append({"ticker": t, "window": k, "start_date": tgt.date.iloc[0], "last_ctx_close": ctx.close.iloc[-1],
                         **{f"pred_{i}": v for i, v in enumerate(p.close.values[:PRED_LEN])},
                         **{f"act_{i}": v for i, v in enumerate(tgt.close.values)}})
        if rows:
            pd.DataFrame(rows).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
        print(f"{t}: +{len(rows)} windows", flush=True)


if __name__ == "__main__":
    main()
