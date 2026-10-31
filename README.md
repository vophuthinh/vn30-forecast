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

### Approach Comparison (30 VN30 Tickers)

| Approach | Avg MAPE ↓ | Avg DirAcc ↑ | Notes |
|---|---|---|---|
| 🥇 **Zero-Shot (ZS)** | **5.01%** | **46.0%** | Best overall — no training needed |
| Log-Return Norm. (LR) | 5.78% | 45.1% | Slight regression from ZS |
| Fine-Tune v1 (FT) | 8.34% | 42.3% | Overfit — hurt 23/30 tickers |

### Multi-Window Walk-Forward (5 Windows × 30 Tickers)

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

### Backtest (VCB, Strategy A vs. Buy & Hold)

| Strategy | Return | Sharpe |
|---|---|---|
| **Kronos Signal (Strategy A)** | **+4.25%** | −0.57 |
| Buy & Hold VCB | −11.15% | — |

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
# Download OHLCV data for all 30 VN30 tickers via yfinance
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
# Walk-forward multi-window evaluation (5 windows × 30 tickers)
python scripts/multiwindow_eval.py

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
│   ├── multiwindow_eval.py           # 5-window walk-forward evaluation
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

- **Source:** Yahoo Finance via `yfinance` (tickers with `.VN` suffix)
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
Train ──────────────────│ Test Window 1 │ Test Window 2 │ ... │ Test Window 5 │
                        ▲               ▲
                   Expanding window — no data leakage
```

- 5 non-overlapping test windows across the full dataset
- Each window: 150 ticker-window pairs evaluated
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

MAPE improves as the model encounters more recent data:

| Window | Avg MAPE |
|---|---|
| W1 (oldest) | 8.73% |
| W2 | 7.89% |
| W3 | 6.41% |
| W4 | 6.12% |
| W5 (most recent) | **5.63%** |

### Notable Tickers

| Ticker | Finding |
|---|---|
| **VIC** | Mean MAPE **41.78%** — major structural price break. Excluded from signal analysis. |
| **FPT** | DirAcc **37%** — reliable contra-signal (flip → ~63% accuracy) |
| **MSN** | MAPE **3.03%** — excellent price accuracy |
| **MBB** | DirAcc **55%**, std=5% — most stable directional signal |
| **VCB** | DirAcc **51%**, std=8.2% — consistent across windows |

---

## 🔬 Fine-Tuning Analysis

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

### Root Causes

- `patience=1` → premature early stopping, model never fully converged
- Joint training on 30 heterogeneous tickers → conflicting gradient updates
- Short regime (3 epochs) → overfit to recent price patterns

### Lesson

> Pre-trained Kronos priors are more valuable than short domain adaptation for VN30. Fine-tuning requires per-ticker or grouped strategies with longer patience.

---

## 📈 Backtest

**Setup:** VCB ticker, Kronos zero-shot directional signals, threshold-based entry/exit (Strategy A).

| Strategy | Return | Sharpe |
|---|---|---|
| **Strategy A (Kronos signal)** | **+4.25%** | −0.57 |
| Buy & Hold VCB | −11.15% | — |

**Outperformance:** ~15.4 pp absolute during the test window.

> The negative Sharpe ratio indicates the outperformance was primarily driven by avoiding a major down-move, not by consistent alpha. Multi-ticker, multi-period backtesting is required to validate robustness.

---

## 🧠 Key Conclusions

1. **Zero-shot wins for VN30.** MAPE=5.01% with no training — pre-trained priors transfer surprisingly well to an emerging market context.

2. **Fine-tuning is a double-edged sword.** v1 (aggressive early stopping, joint training) hurt 23/30 tickers. Careful per-ticker or grouped fine-tuning with patience=3+ is needed.

3. **No ticker is reliably tradeable standalone.** The best directional accuracy (STB at 56%) is just above random. Use Kronos signals as a weak supplementary factor in a broader framework.

4. **VIC is a structural outlier** (mean MAPE 41.78%) due to a corporate restructuring event causing a major price discontinuity. Always inspect for structural breaks before forecasting.

5. **Log-return normalization does not help.** The reconstruction error from log-return→price cancels any benefit from scale normalization.

6. **Backtest shows promise, not proof.** +4.25% vs −11.15% is a strong result on one ticker in one period; systematic validation across all tickers and time periods is the necessary next step.

---

## ⚙️ Requirements

**Python:** 3.11+

**Key dependencies:**

```
torch>=2.1
transformers>=4.38
huggingface_hub
yfinance>=0.2
pandas
numpy
scikit-learn
matplotlib
seaborn
tqdm
safetensors
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
- **Data source:** [yfinance](https://github.com/ranaroussi/yfinance) — Yahoo Finance Python wrapper
- **VN30 index:** Ho Chi Minh Stock Exchange (HOSE) — Vietnam's premier large-cap index

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Made with ❤️ for Vietnamese quantitative finance research
</p>
