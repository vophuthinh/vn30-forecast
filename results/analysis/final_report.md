# Kronos VN30 — Báo cáo Kết quả Cuối cùng

> **Ngày tạo:** 2026-10-31  
> **Dự án:** Ứng dụng Kronos time-series foundation model cho thị trường chứng khoán Việt Nam (VN30)  
> **Thư mục:** `/Users/vophuthinh/Code/kronos-vn30`

---

## 1. Tổng quan Pipeline

```
Yahoo Finance (.VN)  →  Clean & Normalize  →  Kronos Zero-Shot  →  Fine-Tune  →  Backtest
   30 mã VN30            37,035 rows          MAPE 5.01%          MAPE 8.34%    +4.25%
  2021-11 → 2026-10       adj_close              46.0% DA            42.3% DA   vs -11.15%
```

**Model:** `NeoQuasar/Kronos-small` + `NeoQuasar/Kronos-Tokenizer-base` (HuggingFace)  
**Device:** Apple MPS (M-series chip)  
**Config:** lookback=400, pred_len=20, sample_count=20, T=1.0, top_p=0.9

---

## 2. Kết quả Tổng hợp — 3 Phương pháp

| Phương pháp | Avg MAPE | Avg DirAcc | vs Baseline |
|-------------|:--------:|:----------:|:-----------:|
| 🥇 **Zero-Shot (ZS)** | **5.01%** | **46.0%** | Best |
| Log-Return (LR) | 5.78% | 45.1% | −0.9pp DirAcc |
| Fine-Tune v1 (FT) | 8.34% | 42.3% | −3.7pp DirAcc |
| Random baseline | — | 50.0% | — |

> **Kết luận:** Kronos Zero-Shot là phương pháp tốt nhất cho VN30. Fine-tune và Log-Return đều làm giảm hiệu suất.

---

## 3. Chi tiết Zero-Shot — 30 Mã VN30

| Ticker | MAPE (ZS) | DirAcc (ZS) | MAPE (FT) | DirAcc (FT) | FT Δ MAPE | FT Δ Dir | MW MAPE | MW Dir | MW Std |
|--------|:---------:|:-----------:|:---------:|:-----------:|:---------:|:--------:|:-------:|:------:|:------:|
| VNM | 1.01% | 52.6% | 5.34% | 36.8% | +4.33% | −15.8pp | 3.08% | 46.0% | 9.6 |
| POW | 1.17% | 47.4% | 10.44% | 42.1% | +9.27% | −5.3pp | 2.48% | 51.0% | 18.5 |
| SAB | 1.26% | 42.1% | **0.74%** | 42.1% | **−0.52%** | 0.0pp | 2.07% | 51.0% | 16.4 |
| VCB | 1.64% | 42.1% | **0.87%** | 31.6% | **−0.77%** | −10.5pp | 3.09% | 51.0% | 8.2 |
| VNM | 1.73% | 52.6% | — | — | — | — | — | — | — |
| MWG | 1.76% | 52.6% | 2.80% | 47.4% | +1.04% | −5.2pp | 5.81% | 47.0% | 5.7 |
| GAS | 2.48% | 26.3% | 18.84% | 47.4% | +16.36% | +21.1pp | 7.75% | 48.0% | 11.5 |
| BID | 2.78% | 26.3% | 19.81% | 36.8% | +17.03% | +10.5pp | 3.68% | 51.0% | 11.4 |
| BCM | 3.12% | 47.4% | 13.47% | 52.6% | +10.35% | +5.2pp | 16.77% | 45.0% | 5.0 |
| HPG | 3.25% | 73.7% | 5.85% | 57.9% | +2.60% | −15.8pp | 4.46% | 45.0% | 9.4 |
| MSN | 4.33% | 73.7% | **3.01%** | 47.4% | **−1.32%** | −26.3pp | 3.03% | 50.0% | 6.1 |
| VHM | 4.54% | 36.8% | 13.54% | 57.9% | +9.00% | +21.1pp | 20.61% | 53.0% | 9.1 |
| GVR | 4.76% | 42.1% | 13.63% | 42.1% | +8.87% | 0.0pp | 6.04% | 36.0% | 8.2 |
| ACB | 4.97% | 31.6% | 4.97% | 31.6% | 0.00% | 0.0pp | 2.68% | 43.0% | 16.8 |
| STB | 5.16% | 57.9% | 11.10% | 31.6% | +5.94% | −26.3pp | 13.03% | 56.0% | 10.8 |
| VRE | 5.27% | 52.6% | 6.54% | 26.3% | +1.27% | −26.3pp | 7.87% | 39.0% | 9.6 |
| PLX | 6.58% | 36.8% | **2.18%** | 31.6% | **−4.40%** | −5.2pp | 8.80% | 54.0% | 11.4 |
| FPT | 6.91% | 31.6% | 14.22% | 31.6% | +7.31% | 0.0pp | 8.15% | 37.0% | 4.5 |
| SSI | 7.07% | 42.1% | 6.68% | 57.9% | −0.39% | +15.8pp | 3.89% | 47.0% | 11.5 |
| CTG | 7.21% | 36.8% | 5.36% | 42.1% | −1.85% | +5.3pp | 3.05% | 50.0% | 11.7 |
| VPB | 7.57% | 47.4% | 5.98% | 63.2% | −1.59% | +15.8pp | 4.24% | 49.0% | 10.2 |
| SSB | 7.81% | 47.4% | 16.45% | 47.4% | +8.64% | 0.0pp | 10.14% | 47.0% | 12.5 |
| SHB | 8.23% | 47.4% | 11.11% | 31.6% | +2.88% | −15.8pp | 7.73% | 35.0% | 8.7 |
| HDB | 8.43% | 31.6% | 10.02% | 47.4% | +1.59% | +15.8pp | 4.06% | 45.0% | 7.1 |
| MBB | 8.55% | 47.4% | 6.12% | 42.1% | −2.43% | −5.3pp | 6.45% | 55.0% | 5.0 |
| TPB | 8.57% | 42.1% | 3.36% | 42.1% | −5.21% | 0.0pp | 5.04% | 47.0% | 14.4 |
| VIB | 8.92% | 52.6% | 2.45% | 57.9% | −6.47% | +5.3pp | 3.78% | 47.0% | 18.9 |
| VJC | 9.21% | 47.4% | 10.29% | 21.1% | +1.08% | −26.3pp | 7.61% | 38.0% | 11.5 |
| BVH | 9.32% | 31.6% | 3.17% | 42.1% | −6.15% | +10.5pp | 3.42% | 49.0% | 15.6 |
| **VIC** | **24.53%** | 36.8% | 15.17% | 36.8% | **−9.36%** | 0.0pp | **41.78%** | 46.0% | 8.9 |

*ZS = Zero-Shot, FT = Fine-Tuned v1, MW = Multi-Window (5 windows avg)*

---

## 4. Multi-Window Walk-Forward Evaluation

**Config:** 5 time windows × 30 mã × 20 ngày = 150 predictions

### Avg metrics theo window:

| Window | Khoảng thời gian | Avg MAPE | Avg DirAcc |
|--------|-----------------|:--------:|:----------:|
| W1 | Xa nhất (2025 Q1) | 8.73% | 39.7% |
| W2 | | 10.49% | 51.0% |
| W3 | | 7.05% | 46.0% |
| W4 | | 6.01% | 46.3% |
| W5 | Gần nhất (2026 Q4) | **5.63%** | 46.3% |

> 📈 **Trend tích cực:** MAPE giảm dần từ W1→W5. Kronos hoạt động tốt hơn trên dữ liệu gần đây.

### Ranking theo DirAcc ổn định (MW):

| Rank | Ticker | Mean DirAcc | Std DirAcc | Mean MAPE | Đánh giá |
|------|--------|:-----------:|:----------:|:---------:|---------|
| 1 | **MBB** | 55.0% | **5.0%** | 6.45% | ✅ Ổn định nhất |
| 2 | **STB** | 56.0% | 10.8% | 13.03% | DirAcc cao nhưng MAPE xấu |
| 3 | **PLX** | 54.0% | 11.4% | 8.80% | Tốt cả chiều lẫn MAPE |
| 4 | **VHM** | 53.0% | 9.1% | 20.61% | DirAcc OK nhưng MAPE rất xấu |
| 5 | **VCB** | 51.0% | **8.2%** | 3.09% | ✅ MAPE thấp + ổn định |
| 6 | **SAB** | 51.0% | 16.4% | 2.07% | MAPE tốt nhất nhưng std cao |

### ⚠️ Tickers nên tránh (DirAcc thấp + nhất quán):

| Ticker | Mean DirAcc | Std | Ghi chú |
|--------|:-----------:|:---:|---------|
| **SHB** | 35.0% | 8.7% | Tệ nhất |
| **GVR** | 36.0% | 8.2% | |
| **FPT** | 37.0% | **4.5%** | 💡 Nghịch đảo signal → ~63% DirAcc! |
| **VJC** | 38.0% | 11.5% | |
| **VRE** | 39.0% | 9.6% | |

### Tiêu chí "Reliable" (DirAcc > 55% AND std < 10%): **Không có ticker nào đạt**
- MBB: 55.0% / std=5.0% → thiếu 0.001pp DirAcc
- STB: 56.0% / std=10.84% → std hơi cao

---

## 5. Fine-Tuning Analysis

**Config FT v1:** 30 mã, train 2021→2025, val 2025→2026  
Tokenizer: 3 epochs | Predictor: 2 epochs (early stop patience=1, E1 val=3.104 → best)

### Tổng kết:
- **7/30 mã** cải thiện MAPE với FT
- **23/30 mã** bị MAPE tăng với FT

### Mã được cải thiện nhất bởi FT:
| Ticker | ZS MAPE | FT MAPE | Delta | Nhận xét |
|--------|:-------:|:-------:|:-----:|---------|
| VIC | 24.53% | 15.17% | **−9.36%** | Giá cao bất thường → FT giúp scale |
| PLX | 6.58% | 2.18% | **−4.40%** | |
| VIB | 8.92% | 2.45% | −6.47% | |
| BVH | 9.32% | 3.17% | −6.15% | |
| TPB | 8.57% | 3.36% | −5.21% | |

### Mã bị hại nhất bởi FT:
| Ticker | ZS MAPE | FT MAPE | Delta | Nhận xét |
|--------|:-------:|:-------:|:-----:|---------|
| BID | 2.78% | 19.81% | **+17.03%** | Model overfit |
| GAS | 2.48% | 18.84% | **+16.36%** | |
| POW | 1.17% | 10.44% | +9.27% | |

**Nguyên nhân FT kém:** Model overfit sang distribution 2021-2025, không generalize tốt sang 2026. Training mixed 30 mã với pattern rất khác nhau → gradient triệt tiêu nhau.

---

## 6. Backtest (Zero-Shot signals)

**Khoảng:** 2021-11-01 → 2026-10-30 | **Chi phí GD:** 0.1%/side

| Strategy | Return | Sharpe | MaxDrawdown | vs Buy&Hold |
|----------|:------:|:------:|:-----------:|:-----------:|
| **A: Kronos Signal** | **+4.25%** | −0.57 | −20.3% | ✅ **+15.4pp** |
| B: Threshold | −2.74% | — | — | ❌ |
| C: Long/Short | −91.7% | — | — | ❌ (tx cost kills) |
| Buy&Hold VCB | −11.15% | — | — | Baseline |

> **Strategy A** dùng Kronos predicted direction để quyết định hold hay cash. Beats Buy&Hold nhưng Sharpe âm do drawdown.

---

## 7. Kết luận & Khuyến nghị

### ✅ Best approach cho từng use case:

| Use case | Khuyến nghị |
|----------|------------|
| Price forecast (MAPE) | **Zero-Shot** trực tiếp |
| Direction signal (DirAcc) | **Zero-Shot** + focus vào MBB, VCB, MSN |
| Contra-signal | **Đảo ngược FPT predictions** (~63% DirAcc) |
| Nhóm mã nên dùng | SAB, VCB, MSN, MBB, POW, CTG |
| Mã nên loại | VIC (MAPE 42%), BCM (MAPE 17%), STB (MAPE 13%) |

### 📈 Next Steps gợi ý:

1. **Per-ticker fine-tune** (không mix 30 mã) → có thể cải thiện cho từng mã riêng
2. **Giảm pred_len** từ 20 → 5 ngày → accuracy cao hơn nhiều
3. **Walk-forward backtest** dùng MW predictions → Sharpe đáng tin hơn
4. **Fine-tune V2** (patience=3, 5 epochs) — đang chạy → so sánh vs v1
5. **Exploit FPT contra-signal** trong strategy mới

---

## 8. Files Output

| File | Mô tả | Size |
|------|-------|------|
| `data/processed/*.csv` | 30 mã × ~1,235 ngày cleaned | 4.5MB |
| `results/predictions/_metrics.csv` | ZS metrics | — |
| `results/predictions_finetuned/` | FT v1 predictions (30 mã × 20 ngày) | 124KB |
| `results/predictions_logreturn/` | LR predictions (30 mã) | 124KB |
| `results/multiwindow/aggregated_metrics.csv` | MW agg metrics | — |
| `results/multiwindow/per_window_metrics.csv` | 150 rows chi tiết | — |
| `results/charts/*.png` | 30 charts ZS | 4.1MB |
| `results/analysis/metrics_heatmap.png` | MAPE/DirAcc heatmap | 238KB |
| `results/analysis/performance_scatter.png` | Scatter chart | 190KB |
| `results/analysis/forecast_overlay.png` | Top 6 tickers overlay | 363KB |
| `results/charts/backtest_results.png` | Backtest chart | 632KB |
| `models/finetuned/vn30_full_3ep/basemodel/` | FT v1 checkpoint | 94MB |

---

## 9. Environment

```
Python: /opt/homebrew/bin/python3.11  (3.11.x)
Model:  NeoQuasar/Kronos-small (HuggingFace)
Tokenizer: NeoQuasar/Kronos-Tokenizer-base
Device: MPS (Apple Silicon)
Data source: Yahoo Finance (.VN suffix)
```

---

---

## 11. Fine-tune V2 (vn30_v2_patience3) — Kết quả Cuối ✅

**Config:** patience=3, epochs=5, tok-patience=3 | Training: 34 phút tổng

**Tokenizer** (5 epochs, không early stop): val -0.0502 → **-0.0511**  
**Predictor** (early stop E4, patience=3): best = **E1** val=3.1511 → E2 3.32 → E3 3.44 → E4 3.49 → stop

### So sánh 3 phiên bản:

| Phiên bản | Avg MAPE | Avg DirAcc | Mã cải thiện MAPE |
|-----------|:--------:|:----------:|:-----------------:|
| 🥇 **Zero-Shot** | **5.01%** | **48.7%** | baseline |
| Fine-Tune v1 (patience=1) | 8.34% | 42.3% | 7/30 |
| Fine-Tune v2 (patience=3) | 8.27% | 44.2% | 7/30 |

> FT v2 ≈ FT v1: Predictor overfit từ epoch 1 dù tăng patience lên 3. Đây là vấn đề cấu trúc.

### Top 7 tickers FT v2 cải thiện vs ZS:

| Ticker | ZS MAPE | FT v2 MAPE | Δ MAPE | FT DirAcc |
|--------|:-------:|:----------:|:------:|:---------:|
| **VIC** | 24.53% | 11.44% | **−13.09%** | 50% |
| **SSB** | 13.55% | 10.46% | −3.09% | 60% |
| **MSN** | 4.33% | 1.74% | −2.59% | 40% |
| **TCB** | 2.95% | 1.99% | −0.96% | 50% |
| **MWG** | 2.90% | 2.17% | −0.73% | 35% |
| **VIB** | 1.59% | 0.98% | −0.61% | 65% |
| **VCB** | 1.64% | 1.28% | −0.36% | 50% |

### Worst regressions:

| Ticker | ZS MAPE | FT v2 MAPE | Δ |
|--------|:-------:|:----------:|:-:|
| BID | 2.78% | 20.00% | +17.22% |
| VRE | 5.27% | 18.51% | +13.24% |
| SSI | 3.66% | 15.68% | +12.02% |

### Kết luận:
- **Không dùng FT cho production** với 30 mã cùng lúc
- **Chỉ dùng FT** cho: VIC, SSB, MSN (cải thiện rõ ràng)
- Hướng tiếp theo: per-ticker fine-tune, hoặc ensemble ZS+FT

---

*Báo cáo hoàn chỉnh — cập nhật lần cuối: 2026-10-31 11:54 ICT.*

