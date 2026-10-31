# Kronos VN30 — Time-Series Forecasting for Vietnamese Stock Market

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Model-Kronos--small-FF6F00?logo=huggingface&logoColor=white" />
  <img src="https://img.shields.io/badge/Market-VN30-DA291C" />
  <img src="https://img.shields.io/badge/Tickers-30-1565C0" />
  <img src="https://img.shields.io/badge/Status-Research-blueviolet" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

<p align="center">
  Applying <a href="https://huggingface.co/NeoQuasar/Kronos-small"><b>Kronos</b></a> — a pre-trained time-series foundation model (AAAI 2025) — to forecast daily closing prices for all 30 constituents of Vietnam's premier large-cap index.
</p>

---

## Table of Contents

1. [Overview](#-overview)
2. [Key Results](#-key-results)
3. [Quick Start](#-quick-start)
4. [Project Structure](#-project-structure)
5. [Scripts Reference](#-scripts-reference)
6. [Methodology](#-methodology)
7. [Detailed Results](#-detailed-results)
8. [Fine-Tuning Analysis](#-fine-tuning-analysis)
9. [Backtest](#-backtest)
10. [Key Conclusions](#-key-conclusions)
11. [Requirements](#-requirements)
12. [References](#-references)

---

## 🎯 Overview

**Research question:** Can a zero-shot time-series foundation model (Kronos) meaningfully forecast daily closing prices for emerging-market stocks — and does domain fine-tuning help?

**What this project does:**
- Downloads and cleans ~5 years of daily OHLCV data for all 30 VN30 tickers (Yahoo Finance)
- Runs **three inference approaches**: Zero-Shot, Fine-Tuned, and Log-Return normalization
- Conducts a **5-window walk-forward evaluation** (150 ticker-window pairs) for robust out-of-sample assessment
- Implements a **signal-based backtest** vs. Buy & Hold
- Provides a complete set of charts, heatmaps, and analysis reports

**Model:** [`NeoQuasar/Kronos-small`](https://huggingface.co/NeoQuasar/Kronos-small) (~24.7M params, BSQ tokenization) + [`NeoQuasar/Kronos-Tokenizer-base`](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base)

**Data:** Daily OHLCV, 30 VN30 tickers, 2021-11-01 → 2026-10, ~39,000 rows raw / ~37,000 cleaned.

---

## 📊 Key Results

> ⚠️ **Read first — headline result.** Rolling evaluation: 30 tickers × 40 non-overlapping 20-day windows (Aug 2023 – Oct 2026, 1200 pairs; [rolling_report.md](results/analysis/rolling_report.md)). Kronos zero-shot is **worse than a naive random walk**: MAPE 9.01% vs 4.35% (gap +4.66pp, 95% CI [4.14, 5.20]; Kronos wins 27% of pairs). 20-day direction hit 51.6% [48.8, 54.4] ≈ always-up 52.2% ≈ coin flip. No evidence of forecasting skill. The 5.01% / 46.0% figures below come from a single 20-day window (Oct 2026) and are descriptive only.

### Approach Comparison (30 VN30 Tickers) — legacy, single 20-day window (Oct 2026)

| Approach | Avg MAPE ↓ | Avg DirAcc ↑ | Notes |
|---|---|---|---|
| 🥇 **Zero-Shot (ZS)** | **5.01%** | **46.0%** | Best overall — no training needed |
| Log-Return Norm. (LR) | 5.78% | 45.1% | Slight regression from ZS |
| Fine-Tune v1 (FT) | 8.34% | 42.3% | Overfit — hurt 23/30 tickers |

### Legacy 5-Window Evaluation (Jun–Oct 2026, 5 Windows × 30 Tickers)

> Only 19–20 observations per ticker; per-ticker rankings below are noise-level (std error of DirAcc ≈ ±11pp). Prefer the rolling results above.

**Best by MAPE (accuracy):**

| Rank | Ticker | Mean MAPE | Sector |
|---|---|---|---|
| 1 | **SAB** | 2.07% | Consumer Staples |
| 2 | **POW** | 2.48% | Utilities |
| 3 | **ACB** | 2.68% | Banking |
| 4 | **MSN** | 3.03% | Conglomerate |
| 5 | **CTG** | 3.05% | Banking |

**Best by Directional Accuracy (signal quality):**

| Rank | Ticker | Mean DirAcc | Std |
|---|---|---|---|
| 1 | **STB** | 56% | — |
| 2 | **MBB** | 55% | 5% |
| 3 | **PLX** | 54% | — |
| 4 | **VHM** | 53% | — |

> ⚠️ **No ticker met strict reliability criteria (DirAcc > 55% AND std < 10%) simultaneously.** Kronos zero-shot signals should be used as a weak supplementary signal, not a standalone trading system.

### Backtest with real Kronos signals (40 rolling periods, 30 tickers, 0.1%/side cost)

| Strategy | Total return | Ann. Sharpe | Max DD |
|---|---|---|---|
| Equal-weight buy & hold | +43.9% | 0.72 | −15.6% |
| Kronos long-only (hold if predicted 20d return > 0) | +10.6% | 0.57 | −5.1% |
| Kronos long/short (top 5 / bottom 5) | −40.2% | −1.02 | −42.8% |
| 5d-momentum long/short (baseline) | −33.0% | −0.95 | −32.3% |

Kronos long-only trails buy & hold by 0.78pp per period (95% CI [−1.96, +0.44]): not distinguishable from zero; its lower drawdown comes mainly from sitting in cash part of the time. Kronos L/S loses money. Details: [rolling_report.md](results/analysis/rolling_report.md). The earlier single-ticker VCB result (+4.25% vs −11.15%) used a momentum proxy and is kept in [backtest_report.md](results/analysis/backtest_report.md) for reference only.

---

## ⚡ Quick Start

### 1. Clone & Setup

```bash
# Clone this repo
git clone https://github.com/<your-username>/kronos-vn30.git
cd kronos-vn30

# Clone Kronos source (required)
git clone https://github.com/shiyu-coder/Kronos models/Kronos-repo

# Install dependencies
pip install -r requirements.txt
```

### 2. Download & Prepare Data

```bash
# Download OHLCV data for all 30 VN30 tickers from Yahoo Finance (raw HTTP, no yfinance)
python scripts/download_vn30.py

# Clean and preprocess
python scripts/clean_data.py
```

### 3. Run Inference

```bash
# Zero-shot inference (recommended — best results)
python scripts/kronos_inference.py

# Log-return normalization mode
python scripts/kronos_logreturn_inference.py

# Fine-tune first, then run fine-tuned inference
python scripts/finetune_kronos.py --exp-name vn30_ft --epochs 5 --patience 3
python scripts/kronos_inference_finetuned.py --exp-name vn30_ft
```

### 4. Evaluate & Analyze

```bash
# Primary: rolling zero-shot evaluation (1200 pairs, ~90 min on M-series) + baselines + Kronos backtest
python scripts/rolling_eval.py --windows 40
python scripts/rolling_eval.py --windows 40 --ft vn30_v2_patience3 --min-start 2025-11-01   # optional: fine-tuned, out-of-training windows
python scripts/rolling_eval.py --windows 12 --logret --tag lr          # variants (see rolling_report.md section 4)
python scripts/rolling_eval.py --windows 12 --lookback 200 --tag lb200  # also: --temp 0.5 --tag t05, --samples 20 --tag s20
python scripts/finetune_kronos.py --exp-name vn30_v3_cleansplit --epochs 5 --patience 3 --tok-patience 3 --train-end 2025-06-30 --val-end 2025-10-31   # clean-split fine-tune
python scripts/rolling_report.py        # -> results/analysis/rolling_report.md
python tests/test_rolling.py            # sanity checks on rolling outputs

# Legacy: 5-window evaluation and 5-window baseline check
python scripts/multiwindow_eval.py
python scripts/baseline_eval.py

# Generate charts, heatmaps, comparison reports
python scripts/analyze_results.py

# Run backtest
python scripts/backtest.py
```

> **Hardware note:** Developed on Apple Silicon (M-series). MPS acceleration is used automatically. All scripts also run on CPU.

---

## 📁 Project Structure

```
kronos-vn30/
│
├── scripts/                          # All pipeline scripts
│   ├── download_vn30.py              # Fetch OHLCV from Yahoo Finance (.VN suffix)
│   ├── clean_data.py                 # Clean, align, fill gaps, derive `amount`
│   ├── kronos_inference.py           # Zero-shot inference for all 30 tickers
│   ├── kronos_logreturn_inference.py # Log-return normalization inference
│   ├── kronos_inference_finetuned.py # Inference with fine-tuned checkpoint
│   ├── finetune_kronos.py            # Fine-tuning loop (early stopping, patience)
│   ├── rolling_eval.py               # PRIMARY: 40-window rolling zero-shot / FT evaluation, saves paths
│   ├── rolling_report.py             # Baselines, CIs, backtest with real Kronos signals
│   ├── baseline_eval.py              # Naive-baseline check for the legacy 5-window run
│   ├── multiwindow_eval.py           # Legacy 5-window evaluation
│   ├── analyze_results.py            # Charts, heatmaps, comparison reports
│   └── backtest.py                   # Signal-based backtest vs Buy & Hold
│
├── data/
│   ├── raw/                          # Downloaded CSVs (gitignored, ~8MB)
│   └── processed/                    # Cleaned CSVs (gitignored, ~7MB)
│
├── models/
│   ├── Kronos-repo/                  # Kronos source — clone separately (gitignored)
│   └── finetuned/                    # Fine-tune checkpoints (gitignored, ~200MB)
│
├── results/
│   ├── predictions/                  # Zero-shot predictions (30 CSVs)
│   ├── predictions_finetuned/        # Fine-tuned predictions (30 CSVs + metrics)
│   ├── predictions_logreturn/        # Log-return predictions (30 CSVs)
│   ├── multiwindow/                  # Walk-forward metrics (aggregated + per-window)
│   ├── charts/                       # Forecast overlay charts per ticker (PNG)
│   └── analysis/                     # Heatmaps, scatter plots, summary reports
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📜 Scripts Reference

| Script | Description | Key Args |
|---|---|---|
| `download_vn30.py` | Fetch daily OHLCV for all 30 VN30 tickers | `--start`, `--end` |
| `clean_data.py` | Standardize columns, fill gaps, derive `amount` | — |
| `kronos_inference.py` | Zero-shot Kronos inference | `--lookback`, `--pred-len`, `--sample-count` |
| `kronos_logreturn_inference.py` | Log-return normalization inference | same as above |
| `finetune_kronos.py` | Fine-tune predictor + tokenizer | `--exp-name`, `--epochs`, `--patience`, `--tok-patience` |
| `kronos_inference_finetuned.py` | Inference with saved checkpoint | `--exp-name`, `--model-dir`, `--tok-dir` |
| `multiwindow_eval.py` | Walk-forward evaluation | `--windows`, `--pred-len` |
| `analyze_results.py` | Charts & aggregate reports | — |
| `backtest.py` | Backtest signal-based strategy | `--ticker`, `--threshold` |

**Recommended Kronos config:**

```python
lookback      = 400    # context window (trading days)
pred_len      = 20     # forecast horizon
sample_count  = 20     # Monte Carlo samples
temperature   = 1.0
top_p         = 0.9
device        = "mps"  # or "cpu" / "cuda"
```

---

## 🔬 Methodology

### 1. Data Pipeline

- **Source:** Yahoo Finance chart API via `urllib` (tickers with `.VN` suffix)
- **Period:** 2021-11-01 → 2026-10 (~5 years daily)
- **Columns:** `date, open, high, low, close, volume, amount`
- **Cleaning:** forward-fill gaps, remove zero-volume days, align dates

### 2. Inference Approaches

**Zero-Shot (ZS)**
- Feed raw close-price time series directly to `KronosPredictor.predict()`
- No training — leverages pre-trained priors out of the box

**Log-Return Normalization (LR)**
- Convert prices to log-returns before feeding to Kronos
- Reconstruct absolute prices from predicted returns
- Intended to help the model generalize across price scales

**Fine-Tuned (FT)**
- Continue training the Kronos predictor and tokenizer on VN30 data
- Joint training across all 30 tickers simultaneously
- v1: 3 epochs, `patience=1` (aggressive — overfit)
- v2: 5 epochs, `patience=3`, longer warmup (planned improvement)

### 3. Walk-Forward Evaluation

```
[ context: 400 days ][ W1: 20d ]
          [ context: 400 days ][ W2: 20d ]   ... up to W5 (last 20 days)
(zero-shot: model weights never updated)
```

- **Primary evaluation (`rolling_eval.py`):** 40 consecutive non-overlapping 20-day windows per ticker (Aug 2023 – Oct 2026), 1200 pairs, context = preceding 400 days, 10 samples, fixed seeds, predicted paths saved to `results/rolling/paths.csv`
- Legacy `multiwindow_eval.py`: 5 windows, Jun–Oct 2026 only (`multiwindow_eval.WINDOWS`)
- Zero-shot, so no training and no expanding window; every window uses the preceding 400 days as context
- 150 ticker-window pairs in total (30 tickers × 5 windows)
- Metrics: **MAPE** (accuracy), **DirAcc** (directional accuracy), **Std** (consistency)

### 4. Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| MAPE | `mean(|y_pred - y_true| / y_true)` | Lower = more accurate |
| DirAcc | `mean(sign(pred_delta) == sign(true_delta))` | >50% = better than random |
| Std (DirAcc) | std across windows | Lower = more consistent signal |

---

## 🔬 Detailed Results

### Multi-Window MAPE Trend

MAPE by window (naive random walk in brackets). Differences between months reflect market volatility, not model learning:

| Window | Avg MAPE |
|---|---|
| W1 (Jun 2026) | 8.28% (3.15%) |
| W2 | 10.85% (6.99%) |
| W3 | 7.25% (3.55%) |
| W4 | 6.38% (2.97%) |
| W5 (Oct 2026) | 5.14% (3.22%) |

### Notable Tickers

| Ticker | Finding |
|---|---|
| **VIC** | Mean MAPE **41.78%** — price rose ~9× (27k → 241k VND, Jan 2023 → Oct 2026) with no data discontinuity (largest daily move +14%). The model underforecasts a sustained trend far outside its context range. |
| **FPT** | DirAcc **37%** — reliable contra-signal (flip → ~63% accuracy) |
| **MSN** | MAPE **3.03%** — excellent price accuracy |
| **MBB** | DirAcc **55%**, std=5% — most stable directional signal |
| **VCB** | DirAcc **51%**, std=8.2% — consistent across windows |

---

## 🔬 Fine-Tuning Analysis

### Current result (rolling, out-of-training windows, 360 pairs)

| Model | MAPE | Direction hit | vs zero-shot MAPE |
|---|---|---|---|
| Zero-shot | 9.75% | 51.9% | — |
| FT v2 (train ≤ 2025-10, patience 3) | 9.80% | 47.5% | +0.05pp, CI [−0.82, 0.86] |
| FT v3 (clean split: train ≤ 2025-06, val 2025-07→10, test unseen) | 9.02% | 45.8% | −0.73pp, CI [−1.37, −0.12] |
| Naive random walk | 4.82% | always-up 48.1% | — |

v3 gives a small MAPE gain that is statistically detectable but economically irrelevant (still ~2× the naive error), and direction accuracy falls below 50%. Numbers below this table are from the legacy single-window run (v1).

### Where Fine-Tuning Helped (7/30 tickers)

| Ticker | MAPE Δ (pp) |
|---|---|
| VIC | −13.0 |
| SSB | −3.0 |
| MSN | −2.6 |
| TCB | −1.0 |
| MWG | −0.7 |
| VIB | −0.6 |
| VCB | −0.4 |

### Where Fine-Tuning Hurt (23/30 tickers)

| Ticker | MAPE Δ (pp) |
|---|---|
| BID | +17.0 |
| VRE | +13.0 |
| SSI | +12.0 |
| GAS | +9.3 |
| ... | ... |

### Root Causes (hypotheses for v1 only, untested)

> v3 used patience 3 and a clean split and still did not approach the naive baseline, so these causes do not fully explain the result.

- `patience=1` → premature early stopping, model never fully converged
- Joint training on 30 heterogeneous tickers → conflicting gradient updates
- Short regime (3 epochs) → overfit to recent price patterns

### Lesson

> Pre-trained Kronos priors are more valuable than short domain adaptation for VN30. Fine-tuning requires per-ticker or grouped strategies with longer patience.

---

## 📈 Backtest

**Setup:** VCB, long/flat by predicted direction (Strategy A). History 2021–2025 uses a **5-day momentum proxy** as the signal; only the last 20 days use Kronos predictions. The result therefore measures momentum, not Kronos.

| Strategy | Return | Sharpe | Max DD |
|---|---|---|---|
| Strategy A (momentum proxy + Kronos tail) | +4.25% | −0.57 | −20.3% |
| Strategy B (threshold) | −2.74% | −0.88 | −17.0% |
| Strategy C (L/S portfolio, 5 long / 5 short) | −91.71% | −4.16 | −91.9% |
| Buy & Hold VCB | −11.15% | −0.10 | −46.1% |

> All Sharpe ratios are negative. Strategy A's lead over Buy & Hold comes from avoiding a drawdown on one ticker in one period. A valid test needs rolling Kronos signals over the full history.

---

## 🧠 Key Conclusions

1. **Zero-shot is the best of the three Kronos variants** (MAPE 5.01% on the single 20-day Oct-2026 run), but see conclusion 7: it does not beat a naive baseline.

2. **Fine-tuning does not rescue Kronos.** v1 hurt 23/30 tickers on one window. With patience 3 and a clean train/val/test split (v3), MAPE improves 0.73pp (9.75 → 9.02%) but stays far above naive (4.82%) and direction hit is 45.8%. Per-ticker / grouped fine-tuning was not tried.

3. **No ticker is reliably tradeable standalone.** The best directional accuracy (STB at 56%) is just above random. Use Kronos signals as a weak supplementary factor in a broader framework.

4. **VIC is a trend outlier** (mean MAPE 41.78%): a ~9× rally with no data break (checked in `data/processed/VIC.csv`). Earlier "corporate restructuring" explanation had no evidence and is withdrawn.

5. **Variants shift MAPE but none beats naive** (360 pairs, same windows; [rolling_report.md §4](results/analysis/rolling_report.md)): log-return input 8.08% (−1.67pp vs 9.75%), lookback 200 6.96% (−2.80pp), temperature 0.5 9.07%, lookback 100 9.31% (not significant), 20 samples 9.76% (no effect). The lookback gain is not monotonic (100 ≈ default, 200 best), which suggests noise or a context-length quirk rather than a tuning direction. These were tried on the same test windows, so treat as exploratory. The earlier claim that log-return normalization hurts came from one window and is withdrawn.

6. **Kronos signals do not beat buy & hold** in a 40-period backtest with real predictions: long-only +10.6% vs +43.9%, long/short −40.2%.

7. **Kronos zero-shot loses to a naive random walk** on 1200 rolling pairs (MAPE 9.01% vs 4.35%, direction hit 51.6% ≈ always-up 52.2%). Earlier "zero-shot wins" applies only among the three Kronos variants.

8. **Best case still loses to naive.** Best variant (lookback 200) MAPE 6.96% vs naive 4.82%; no variant has direction hit clearly above 50%. Forecasting daily-close paths 20 days ahead with this model on VN30 shows no demonstrated skill.

---

## ⚙️ Requirements

**Python:** 3.11+

**Key dependencies:**

Pinned in `requirements.txt` (the versions used to produce the results):

```
torch, transformers, huggingface_hub, safetensors, einops,
pandas, numpy, scipy, matplotlib, seaborn
```

Install all:

```bash
pip install -r requirements.txt
```

**Models** (auto-downloaded from HuggingFace Hub on first run):

| Component | HuggingFace ID |
|---|---|
| Main model | [`NeoQuasar/Kronos-small`](https://huggingface.co/NeoQuasar/Kronos-small) |
| Tokenizer | [`NeoQuasar/Kronos-Tokenizer-base`](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base) |

**Hardware:**
- Developed and tested on **Apple Silicon (M-series)** — MPS acceleration used automatically
- Also works on CPU (slower) and CUDA GPU
- Zero-shot inference for 30 tickers: ~5–10 minutes on M-series

---

## 📚 References

- **Kronos paper:** *Kronos: A Universal Time Series Foundation Model* — Shiyu Wang et al., AAAI 2025
- **Kronos GitHub:** [https://github.com/shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)
- **HuggingFace:** [NeoQuasar/Kronos-small](https://huggingface.co/NeoQuasar/Kronos-small)
- **Data source:** Yahoo Finance chart API
- **VN30 index:** Ho Chi Minh Stock Exchange (HOSE) — Vietnam's premier large-cap index

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Made with ❤️ for Vietnamese quantitative finance research
</p>
