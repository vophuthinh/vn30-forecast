#!/usr/bin/env python3
"""LEGACY: Strategy A/B/C use a 5-day momentum proxy for 2021-2025; only the last 20 days use Kronos signals.
For Kronos-signal backtests see scripts/rolling_report.py.

Kronos VN30 Walk-Forward Backtesting Engine
============================================
- Strategy A: Kronos Signal (full capital, from momentum proxy on history + Oct-26 actuals)
- Strategy B: Kronos + Threshold (only trade if |predicted_return| > 1%)
- Strategy C: Kronos Portfolio (long top-5 / short bottom-5 across all 30 tickers)

For the 20-day Oct-2026 window  -> uses actual Kronos predictions.
For the longer equity curve (2021-2026) -> uses 5-day momentum as Kronos proxy.

Transaction costs: 0.1% per side  (buy + sell = 0.2% round-trip)
Risk-free rate: 5% p.a.
"""

import os
import sys
import warnings
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path("/Users/vophuthinh/Code/kronos-vn30")
PROCESSED   = ROOT / "data" / "processed"
PRED_DIR    = ROOT / "results" / "predictions"
CHARTS_DIR  = ROOT / "results" / "charts"
ANALYSIS_DIR= ROOT / "results" / "analysis"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

TC          = 0.001          # 0.1% one-way transaction cost
RF_DAILY    = 0.05 / 252     # 5% annual risk-free rate
TRADING_DAYS= 252

VN30_TICKERS = [
    "ACB","BCM","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG",
    "MBB","MSN","MWG","PLX","POW","SAB","SHB","SSB","SSI","STB",
    "TCB","TPB","VCB","VHM","VIB","VIC","VJC","VNM","VPB","VRE",
]

# ── helpers ────────────────────────────────────────────────────────────────────

def load_processed(ticker: str) -> pd.DataFrame:
    path = PROCESSED / f"{ticker}.csv"
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = df[["date","open","high","low","close","volume"]].copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    return df


def load_prediction(ticker: str) -> pd.DataFrame | None:
    path = PRED_DIR / f"{ticker}_pred.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df


def compute_metrics(equity: pd.Series, name: str = "") -> dict:
    """Compute performance metrics from a daily equity curve (indexed by date)."""
    ret = equity.pct_change().dropna()
    ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
    if len(ret) == 0 or equity.iloc[0] == 0:
        return dict(name=name, total_return=0, sharpe=0, max_dd=0, win_rate=0, profit_factor=0)

    equity = equity.dropna()
    if len(equity) < 2:
        return dict(name=name, total_return=0, sharpe=0, max_dd=0, win_rate=0, profit_factor=0)
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

    excess = ret - RF_DAILY
    sharpe = (excess.mean() / ret.std() * np.sqrt(TRADING_DAYS)) if ret.std() > 0 else 0.0

    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    max_dd = dd.min() * 100

    wins  = ret[ret > 0]
    loses = ret[ret < 0]
    win_rate = len(wins) / len(ret) * 100 if len(ret) > 0 else 0
    gross_profit = wins.sum()
    gross_loss   = abs(loses.sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf

    return dict(
        name=name,
        total_return=round(total_return, 2),
        sharpe=round(sharpe, 3),
        max_dd=round(max_dd, 2),
        win_rate=round(win_rate, 2),
        profit_factor=round(profit_factor, 3),
    )


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY A & B — Single-ticker, long-only, uses full historical data
# Signal: 5-day momentum proxy on 2021-2025; Kronos actual on Oct-2026 holdout
# ══════════════════════════════════════════════════════════════════════════════

def run_strategy_ab_ticker(ticker: str):
    """
    Returns (equity_A, equity_B, per_trade_df) for one ticker.
    Equity series is indexed by date, starting at 1.0
    """
    hist = load_processed(ticker)
    pred = load_prediction(ticker)

    if hist is None or len(hist) < 50:
        return None, None, None

    # Build a unified signal series:
    # Part 1: 2021 onwards (momentum proxy — 5-day return direction)
    hist = hist.copy()
    hist = hist[hist["date"] >= pd.Timestamp("2021-01-01")].reset_index(drop=True)
    hist["mom5"] = hist["close"].pct_change(5).shift(1)   # yesterday's 5-day momentum → today's signal

    # Part 2: Oct-2026 holdout — replace momentum signal with Kronos signal
    if pred is not None and len(pred) >= 2:
        pred = pred.copy()
        pred["pred_ret"] = pred["close"].pct_change()
        kronos_dates = pred["date"].tolist()
        kronos_signal = dict(zip(pred["date"], pred["pred_ret"]))
    else:
        kronos_dates   = []
        kronos_signal  = {}

    # Build daily signal frame
    records = []
    capital_a = 1.0
    capital_b = 1.0
    in_pos_a  = False
    in_pos_b  = False
    entry_a   = entry_b = 0.0

    trades_a = []
    trades_b = []

    prev_close = None

    for i, row in hist.iterrows():
        date  = row["date"]
        close = row["close"]
        if pd.isna(close) or close <= 0:
            prev_close = close
            records.append(dict(date=date, equity_a=capital_a, equity_b=capital_b))
            continue

        # Determine signal
        if date in kronos_signal:
            signal_val = kronos_signal[date]
        else:
            signal_val = row["mom5"] if not pd.isna(row["mom5"]) else 0.0

        sig_a = 1 if signal_val > 0 else -1   # BUY or SELL
        sig_b = (1 if signal_val >  0.01 else
                (-1 if signal_val < -0.01 else  0))   # threshold 1%

        # ── Strategy A ──────────────────────────
        if sig_a == 1 and not in_pos_a:
            # Enter long
            entry_a   = close * (1 + TC)
            in_pos_a  = True
        elif sig_a == -1 and in_pos_a:
            # Exit long
            exit_price = close * (1 - TC)
            ret = exit_price / entry_a - 1
            capital_a *= (1 + ret)
            trades_a.append(ret)
            in_pos_a = False

        # ── Strategy B ──────────────────────────
        if sig_b == 1 and not in_pos_b:
            entry_b   = close * (1 + TC)
            in_pos_b  = True
        elif sig_b == -1 and in_pos_b:
            exit_price = close * (1 - TC)
            ret = exit_price / entry_b - 1
            capital_b *= (1 + ret)
            trades_b.append(ret)
            in_pos_b = False
        elif sig_b == 0 and in_pos_b:
            # Exit on neutral signal (hold cash)
            exit_price = close * (1 - TC)
            ret = exit_price / entry_b - 1
            capital_b *= (1 + ret)
            trades_b.append(ret)
            in_pos_b = False

        records.append(dict(date=date, equity_a=capital_a, equity_b=capital_b))
        prev_close = close

    # Close any open positions at end
    if in_pos_a and prev_close:
        exit_price = prev_close * (1 - TC)
        ret = exit_price / entry_a - 1
        capital_a *= (1 + ret)
        trades_a.append(ret)
    if in_pos_b and prev_close:
        exit_price = prev_close * (1 - TC)
        ret = exit_price / entry_b - 1
        capital_b *= (1 + ret)
        trades_b.append(ret)

    df_eq = pd.DataFrame(records).set_index("date")
    equity_a = df_eq["equity_a"]
    equity_b = df_eq["equity_b"]

    # Normalise to start at 1.0
    equity_a = equity_a / equity_a.iloc[0]
    equity_b = equity_b / equity_b.iloc[0]

    return equity_a, equity_b, trades_a


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY C — Portfolio: long top-5 / short bottom-5 daily
# Uses Oct-2026 predictions for ranking + actual prices for P&L
# Also builds a longer history using 5-day momentum for ranking (proxy)
# ══════════════════════════════════════════════════════════════════════════════

def run_strategy_c() -> tuple[pd.Series, dict]:
    """
    Returns (equity_c, per_ticker_pnl_dict)
    """
    # Load all historical data
    hist_data = {}
    for t in VN30_TICKERS:
        try:
            df = load_processed(t)
            df = df[df["date"] >= pd.Timestamp("2021-01-01")].reset_index(drop=True)
            hist_data[t] = df.set_index("date")["close"]
        except Exception:
            pass

    # Load Kronos predictions (Oct-2026)
    pred_data = {}
    for t in VN30_TICKERS:
        p = load_prediction(t)
        if p is not None and len(p) >= 2:
            p["pred_ret"] = p["close"].pct_change()
            pred_data[t] = p.set_index("date")["pred_ret"]

    # Build a combined price DataFrame
    price_df = pd.DataFrame(hist_data).sort_index()

    # Build signal DataFrame
    # For each date, signal = 5-day momentum (proxy for non-holdout days)
    # For Oct-2026 holdout dates, use Kronos predicted return
    mom5_df = price_df.pct_change(5).shift(1)

    # Oct-2026 Kronos signals
    kronos_df = pd.DataFrame(pred_data).sort_index()

    # Merge: where kronos available, use it; else use momentum
    signal_df = mom5_df.copy()
    for date in kronos_df.index:
        if date in signal_df.index:
            for t in VN30_TICKERS:
                if t in kronos_df.columns and not pd.isna(kronos_df.loc[date, t]):
                    signal_df.loc[date, t] = kronos_df.loc[date, t]

    # Daily actual returns
    actual_ret_df = price_df.pct_change()

    # Walk through each day
    dates = sorted(set(price_df.index) & set(signal_df.index))
    dates = [d for d in dates if d >= pd.Timestamp("2021-06-01")]  # need some warmup

    capital     = 1.0
    equity_vals = [capital]
    equity_dates= [dates[0]] if dates else []

    # Track per-ticker cumulative PnL
    ticker_pnl = {t: 0.0 for t in VN30_TICKERS}

    for i in range(1, len(dates)):
        date     = dates[i]
        prev_date= dates[i-1]

        # Signals as of previous day close (predict today's direction)
        sig_row = signal_df.loc[prev_date] if prev_date in signal_df.index else pd.Series(dtype=float)
        sig_row = sig_row.dropna()

        if len(sig_row) < 10:
            equity_vals.append(capital)
            equity_dates.append(date)
            continue

        ranked = sig_row.sort_values(ascending=False)
        longs  = ranked.head(5).index.tolist()
        shorts = ranked.tail(5).index.tolist()

        # Actual returns today
        act_row = actual_ret_df.loc[date] if date in actual_ret_df.index else pd.Series(dtype=float)

        # Long side: equal weight
        long_ret  = np.nanmean([act_row.get(t, 0.0) for t in longs])
        short_ret = np.nanmean([act_row.get(t, 0.0) for t in shorts])

        # Long-short return (both sides use 50% capital each)
        daily_ret = 0.5 * (long_ret - short_ret) - 2 * TC  # 4 round-trip costs amortised

        if np.isnan(daily_ret) or np.isinf(daily_ret):
            equity_vals.append(capital)
            equity_dates.append(date)
            continue
        capital *= (1 + daily_ret)
        equity_vals.append(capital)
        equity_dates.append(date)

        # Track per-ticker contribution
        for t in longs:
            contrib = act_row.get(t, 0.0)
            if not np.isnan(contrib):
                ticker_pnl[t] += 0.5 / 5 * contrib
        for t in shorts:
            contrib = act_row.get(t, 0.0)
            if not np.isnan(contrib):
                ticker_pnl[t] -= 0.5 / 5 * contrib

    equity_c = pd.Series(equity_vals, index=equity_dates)
    if len(equity_c) > 0:
        equity_c = equity_c / equity_c.iloc[0]

    return equity_c, ticker_pnl


# ══════════════════════════════════════════════════════════════════════════════
# BUY & HOLD BENCHMARK (VCB)
# ══════════════════════════════════════════════════════════════════════════════

def buy_and_hold_vcb() -> pd.Series:
    df = load_processed("VCB")
    df = df[df["date"] >= pd.Timestamp("2021-01-01")].reset_index(drop=True)
    df = df.set_index("date")["close"]
    bh = df / df.iloc[0]
    return bh


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(
    equity_a: pd.Series,
    equity_b: pd.Series,
    equity_c: pd.Series,
    bh:       pd.Series,
    ticker_pnl: dict,
):
    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.42, wspace=0.30)

    COLORS = {
        "A":  "#00d4ff",
        "B":  "#ff6b35",
        "C":  "#7fff00",
        "BH": "#aaaaaa",
    }

    ax_style = dict(facecolor="#161b22", labelcolor="#cccccc",
                    tick_params=dict(colors="#aaaaaa"))

    def style_ax(ax, title=""):
        ax.set_facecolor("#161b22")
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=8)
        ax.tick_params(colors="#aaaaaa")
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.grid(True, color="#21262d", linewidth=0.5, linestyle="--")

    # ── Panel 1: Equity Curves ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1, "📈  Equity Curves — Strategy A, B, C vs Buy & Hold (VCB)")

    def align_and_plot(ax, eq, label, color, lw=1.5, ls="-"):
        if len(eq) == 0:
            return
        eq = eq.sort_index()
        ax.plot(eq.index, eq.values, label=label, color=color, linewidth=lw, linestyle=ls)

    align_and_plot(ax1, equity_a, "Strategy A (Kronos Signal)",             COLORS["A"])
    align_and_plot(ax1, equity_b, "Strategy B (Kronos + Threshold)",         COLORS["B"])
    align_and_plot(ax1, equity_c, "Strategy C (Portfolio L/S)",              COLORS["C"])
    align_and_plot(ax1, bh,       "Buy & Hold VCB",                          COLORS["BH"], lw=1.2, ls="--")

    ax1.set_ylabel("Portfolio Value (normalised)", fontsize=10)
    ax1.legend(framealpha=0.3, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="white", fontsize=9)

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    style_ax(ax2, "📉  Maximum Drawdown")

    def plot_dd(ax, eq, label, color):
        if len(eq) == 0:
            return
        eq = eq.sort_index()
        roll_max = eq.cummax()
        dd = (eq - roll_max) / roll_max * 100
        ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color=color, label=label)
        ax.plot(dd.index, dd.values, color=color, linewidth=0.8)

    plot_dd(ax2, equity_a, "Strategy A", COLORS["A"])
    plot_dd(ax2, equity_b, "Strategy B", COLORS["B"])
    plot_dd(ax2, equity_c, "Strategy C", COLORS["C"])
    plot_dd(ax2, bh,       "Buy & Hold", COLORS["BH"])

    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.legend(framealpha=0.3, facecolor="#21262d", edgecolor="#30363d",
               labelcolor="white", fontsize=9)

    # ── Panel 3: Monthly Returns Heatmap ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, :])
    style_ax(ax3, "🗓  Monthly Returns Heatmap (%) — 3 Strategies")

    def monthly_returns(eq: pd.Series, label: str) -> pd.Series:
        if len(eq) == 0:
            return pd.Series(dtype=float)
        eq = eq.sort_index()
        monthly = eq.resample("ME").last()
        ret = monthly.pct_change().dropna() * 100
        ret.index = ret.index.to_period("M")
        return ret

    mr_a = monthly_returns(equity_a, "A")
    mr_b = monthly_returns(equity_b, "B")
    mr_c = monthly_returns(equity_c, "C")

    # Build combined heatmap — last 18 months × 3 strategies
    all_periods = sorted(set(mr_a.index) | set(mr_b.index) | set(mr_c.index))
    if len(all_periods) > 18:
        all_periods = all_periods[-18:]

    heat_data = np.zeros((3, len(all_periods)))
    for j, p in enumerate(all_periods):
        heat_data[0, j] = mr_a.get(p, np.nan)
        heat_data[1, j] = mr_b.get(p, np.nan)
        heat_data[2, j] = mr_c.get(p, np.nan)

    heat_data = np.nan_to_num(heat_data, nan=0.0)

    cmap = LinearSegmentedColormap.from_list(
        "rg", ["#cf1b2b","#161b22","#21a552"], N=256
    )
    vmax = max(abs(heat_data.max()), abs(heat_data.min()), 0.01)
    im = ax3.imshow(heat_data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(["Strategy A", "Strategy B", "Strategy C"], color="white", fontsize=9)
    ax3.set_xticks(range(len(all_periods)))
    ax3.set_xticklabels([str(p) for p in all_periods], rotation=45, ha="right",
                         color="#aaaaaa", fontsize=7)

    for i in range(3):
        for j in range(len(all_periods)):
            val = heat_data[i, j]
            txt_color = "white" if abs(val) < vmax * 0.6 else ("black" if val > 0 else "white")
            ax3.text(j, i, f"{val:.1f}%", ha="center", va="center",
                     fontsize=6.5, color=txt_color)

    plt.colorbar(im, ax=ax3, shrink=0.8, pad=0.01).ax.tick_params(labelcolor="#aaaaaa")

    # ── Panel 4: Per-Ticker PnL (Portfolio Strategy C) ───────────────────────
    ax4 = fig.add_subplot(gs[3, :])
    style_ax(ax4, "💹  Per-Ticker P&L Contribution — Strategy C (Top 10 + Bottom 10)")

    pnl_series = pd.Series(ticker_pnl).sort_values()
    show = pd.concat([pnl_series.head(10), pnl_series.tail(10)]).drop_duplicates()
    show = show.sort_values()

    bar_colors = [COLORS["A"] if v >= 0 else COLORS["B"] for v in show.values]
    bars = ax4.barh(show.index, show.values * 100, color=bar_colors, edgecolor="none", height=0.65)
    ax4.axvline(0, color="#555555", linewidth=0.8)
    ax4.set_xlabel("Cumulative P&L Contribution (%)", fontsize=9)

    for bar, val in zip(bars, show.values * 100):
        x = bar.get_width()
        ax4.text(x + (0.05 if x >= 0 else -0.05), bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}%", va="center", ha="left" if x >= 0 else "right",
                 color="white", fontsize=7.5)

    fig.suptitle("Kronos VN30 — Backtesting Engine Results", color="white",
                 fontsize=16, fontweight="bold", y=0.995)

    out = CHARTS_DIR / "backtest_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[✓] Chart saved → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════

def write_report(metrics_list: list[dict], ticker_pnl: dict, chart_path: Path):
    # Sort tickers by PnL (filter NaN)
    clean_pnl = {k: v for k, v in ticker_pnl.items() if not np.isnan(v)}
    sorted_tickers = sorted(clean_pnl.items(), key=lambda x: x[1], reverse=True)
    top5   = sorted_tickers[:5]
    worst5 = sorted_tickers[-5:]

    bh_m = next((m for m in metrics_list if m["name"] == "Buy&Hold (VCB)"), {})
    strat_metrics = [m for m in metrics_list if m["name"] != "Buy&Hold (VCB)"]

    lines = [
        "# Kronos VN30 — Backtesting Report\n",
        f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
        "\n---\n",
        "## 1. Strategy Performance vs Buy & Hold\n",
        "| Strategy | Total Return | Sharpe Ratio | Max Drawdown | Win Rate | Profit Factor |",
        "|----------|-------------|-------------|-------------|---------|--------------|",
    ]

    all_m = strat_metrics + [bh_m]
    for m in all_m:
        pf = f"{m.get('profit_factor','—'):.3f}" if isinstance(m.get("profit_factor"), float) else "—"
        lines.append(
            f"| {m.get('name','—')} "
            f"| {m.get('total_return','—'):.2f}% "
            f"| {m.get('sharpe','—'):.3f} "
            f"| {m.get('max_dd','—'):.2f}% "
            f"| {m.get('win_rate','—'):.2f}% "
            f"| {pf} |"
        )

    lines += [
        "\n---\n",
        "## 2. Top 5 Performers — Portfolio Strategy C\n",
        "| Rank | Ticker | Cumulative Contribution |",
        "|------|--------|------------------------|",
    ]
    for rank, (t, pnl) in enumerate(top5, 1):
        lines.append(f"| {rank} | **{t}** | {pnl*100:.2f}% |")

    lines += [
        "\n---\n",
        "## 3. Bottom 5 Performers — Portfolio Strategy C\n",
        "| Rank | Ticker | Cumulative Contribution |",
        "|------|--------|------------------------|",
    ]
    for rank, (t, pnl) in enumerate(worst5, 1):
        lines.append(f"| {rank} | **{t}** | {pnl*100:.2f}% |")

    # Find best strategy by Sharpe
    best = max(strat_metrics, key=lambda m: m.get("sharpe", -999))

    lines += [
        "\n---\n",
        "## 4. Analysis\n",
        f"### Best Strategy: **{best.get('name')}**\n",
        f"The best risk-adjusted strategy by Sharpe Ratio is **{best.get('name')}** "
        f"(Sharpe = {best.get('sharpe','—'):.3f}). "
        "Below is a breakdown:\n",
        "- **Strategy A (Kronos Signal)** trades every day based on the predicted direction "
          "of the next close price. It is fully invested at all times and captures both "
          "trending and mean-reverting moves. High turnover increases transaction costs.\n",
        "- **Strategy B (Kronos + Threshold)** filters out low-conviction signals "
          "(|predicted return| ≤ 1%), staying in cash on ambiguous days. "
          "This reduces the number of round-trips and typically improves the Sharpe Ratio "
          "at the cost of missing some gains.\n",
        "- **Strategy C (Long/Short Portfolio)** is market-neutral: each day it goes long "
          "the 5 highest-ranked tickers and short the 5 lowest-ranked. "
          "Returns are driven by the spread between winners and losers, making it less "
          "sensitive to broad VN30 index direction.\n",
        "\n---\n",
        "## 5. Disclaimer\n",
        "> **⚠️ Important**: These results are based on a **zero-shot** application of the "
        "Kronos-small time-series foundation model trained on international equity data. "
        "The model has **not** been fine-tuned on Vietnamese market data. "
        "For the 20-day Oct-2026 holdout window, actual Kronos predictions are used. "
        "For the longer historical equity curves, a **5-day momentum proxy** is used as a "
        "stand-in for Kronos signals. "
        "Past simulated performance does **not** guarantee future results. "
        "This analysis is for research purposes only and does not constitute investment advice.\n",
    ]

    out = ANALYSIS_DIR / "backtest_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[✓] Report saved → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("  Kronos VN30 Walk-Forward Backtest Engine")
    print("=" * 64)

    # ── Strategy A & B: aggregate across VN30 ──────────────────────────────
    print("\n[1/3] Running Strategy A & B (per-ticker, aggregated)…")
    all_a, all_b = [], []
    for ticker in VN30_TICKERS:
        try:
            ea, eb, _ = run_strategy_ab_ticker(ticker)
            if ea is not None and len(ea) > 10:
                all_a.append(ea)
                all_b.append(eb)
        except Exception as e:
            print(f"  [!] {ticker}: {e}")

    if not all_a:
        print("[ERROR] No data available for Strategy A/B")
        sys.exit(1)

    # Align all series to common dates and average (equal-weight portfolio of single-ticker strategies)
    def avg_equity(series_list: list[pd.Series]) -> pd.Series:
        df = pd.DataFrame({i: s for i, s in enumerate(series_list)})
        df = df.sort_index().ffill()
        return df.mean(axis=1)

    equity_a = avg_equity(all_a)
    equity_b = avg_equity(all_b)

    # ── Strategy C ─────────────────────────────────────────────────────────
    print("[2/3] Running Strategy C (long/short portfolio)…")
    try:
        equity_c, ticker_pnl = run_strategy_c()
    except Exception as e:
        print(f"  [!] Strategy C failed: {e}")
        traceback.print_exc()
        equity_c  = pd.Series(dtype=float)
        ticker_pnl = {t: 0.0 for t in VN30_TICKERS}

    # ── Buy & Hold benchmark ───────────────────────────────────────────────
    bh = buy_and_hold_vcb()

    # ── Metrics ────────────────────────────────────────────────────────────
    print("[3/3] Computing metrics…")
    m_a  = compute_metrics(equity_a, "Strategy A (Kronos Signal)")
    m_b  = compute_metrics(equity_b, "Strategy B (Kronos + Threshold)")
    m_c  = compute_metrics(equity_c, "Strategy C (Portfolio L/S)")
    m_bh = compute_metrics(bh,       "Buy&Hold (VCB)")
    metrics_list = [m_a, m_b, m_c, m_bh]

    print("\n── Performance Summary ─────────────────────────────────────")
    print(f"{'Strategy':<35} {'TotalRet':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8}")
    print("-" * 70)
    for m in metrics_list:
        print(f"{m['name']:<35} {m['total_return']:>9.2f}% {m['sharpe']:>8.3f} "
              f"{m['max_dd']:>7.2f}% {m['win_rate']:>7.2f}%")

    # ── Top 3 tickers in portfolio ─────────────────────────────────────────
    top3 = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)[:3]
    print("\n── Top 3 Tickers (Strategy C) ──────────────────────────────")
    for rank, (t, pnl) in enumerate(top3, 1):
        print(f"  {rank}. {t}: {pnl*100:.2f}%")

    # ── Plots & Report ─────────────────────────────────────────────────────
    chart_path  = plot_results(equity_a, equity_b, equity_c, bh, ticker_pnl)
    report_path = write_report(metrics_list, ticker_pnl, chart_path)

    print("\n" + "=" * 64)
    print("  Done!")
    print(f"  Chart  → {chart_path}")
    print(f"  Report → {report_path}")
    print("=" * 64)

    # Return structured results for caller
    return {
        "strategy_a": m_a,
        "strategy_b": m_b,
        "strategy_c": m_c,
        "buy_hold":   m_bh,
        "top3_tickers": top3,
        "chart":  str(chart_path),
        "report": str(report_path),
    }


if __name__ == "__main__":
    results = main()
