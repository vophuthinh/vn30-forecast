# Kronos VN30 — Backtesting Report

*Generated: 2026-10-30 23:00:35*


---

## 1. Strategy Performance vs Buy & Hold

| Strategy | Total Return | Sharpe Ratio | Max Drawdown | Win Rate | Profit Factor |
|----------|-------------|-------------|-------------|---------|--------------|
| Strategy A (Kronos Signal) | 4.25% | -0.571 | -20.34% | 32.39% | 1.042 |
| Strategy B (Kronos + Threshold) | -2.74% | -0.877 | -17.00% | 32.63% | 0.984 |
| Strategy C (Portfolio L/S) | -91.71% | -4.160 | -91.92% | 35.22% | 0.519 |
| Buy&Hold (VCB) | -11.15% | -0.102 | -46.10% | 44.92% | 1.016 |

---

## 2. Top 5 Performers — Portfolio Strategy C

| Rank | Ticker | Cumulative Contribution |
|------|--------|------------------------|
| 1 | **VIC** | 15.91% |
| 2 | **SSI** | 5.40% |
| 3 | **SSB** | 5.24% |
| 4 | **VRE** | 4.85% |
| 5 | **VHM** | 4.26% |

---

## 3. Bottom 5 Performers — Portfolio Strategy C

| Rank | Ticker | Cumulative Contribution |
|------|--------|------------------------|
| 1 | **VCB** | -4.44% |
| 2 | **GVR** | -5.70% |
| 3 | **SHB** | -6.64% |
| 4 | **GAS** | -7.17% |
| 5 | **MWG** | -11.10% |

---

## 4. Analysis

### Best Strategy: **Strategy A (Kronos Signal)**

The best risk-adjusted strategy by Sharpe Ratio is **Strategy A (Kronos Signal)** (Sharpe = -0.571). Below is a breakdown:

- **Strategy A (Kronos Signal)** trades every day based on the predicted direction of the next close price. It is fully invested at all times and captures both trending and mean-reverting moves. High turnover increases transaction costs.

- **Strategy B (Kronos + Threshold)** filters out low-conviction signals (|predicted return| ≤ 1%), staying in cash on ambiguous days. This reduces the number of round-trips and typically improves the Sharpe Ratio at the cost of missing some gains.

- **Strategy C (Long/Short Portfolio)** is market-neutral: each day it goes long the 5 highest-ranked tickers and short the 5 lowest-ranked. Returns are driven by the spread between winners and losers, making it less sensitive to broad VN30 index direction.


---

## 5. Disclaimer

> **⚠️ Important**: These results are based on a **zero-shot** application of the Kronos-small time-series foundation model trained on international equity data. The model has **not** been fine-tuned on Vietnamese market data. For the 20-day Oct-2026 holdout window, actual Kronos predictions are used. For the longer historical equity curves, a **5-day momentum proxy** is used as a stand-in for Kronos signals. Past simulated performance does **not** guarantee future results. This analysis is for research purposes only and does not constitute investment advice.
