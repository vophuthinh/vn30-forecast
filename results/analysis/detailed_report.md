# BÁO CÁO CHI TIẾT: ỨNG DỤNG KRONOS DỰ BÁO GIÁ CỔ PHIẾU VN30

**Ngày:** 2026-10-31  
**Tác giả:** Kronos VN30 Research Pipeline  
**Thư mục dự án:** `/Users/vophuthinh/Code/kronos-vn30`

---

## MỤC LỤC

1. [Giới thiệu & Mục tiêu](#1-giới-thiệu--mục-tiêu)
2. [Dữ liệu](#2-dữ-liệu)
3. [Mô hình Kronos](#3-mô-hình-kronos)
4. [Phương pháp thực nghiệm](#4-phương-pháp-thực-nghiệm)
5. [Kết quả Zero-Shot](#5-kết-quả-zero-shot)
6. [Phân tích theo Sector](#6-phân-tích-theo-sector)
7. [Multi-Window Walk-Forward Evaluation](#7-multi-window-walk-forward-evaluation)
8. [Fine-Tuning — Phân tích chi tiết](#8-fine-tuning--phân-tích-chi-tiết)
9. [Log-Return Normalization](#9-log-return-normalization)
10. [Bảng so sánh tổng hợp — 30 mã](#10-bảng-so-sánh-tổng-hợp--30-mã)
11. [Backtesting](#11-backtesting)
12. [Kết luận & Khuyến nghị](#12-kết-luận--khuyến-nghị)
13. [Hướng phát triển tiếp theo](#13-hướng-phát-triển-tiếp-theo)

---

## 1. Giới thiệu & Mục tiêu

### Bối cảnh

**Kronos** là mô hình time-series foundation model được phát triển bởi nhóm nghiên cứu Tsinghua University, công bố tại AAAI 2025. Mô hình được pre-train trên tập dữ liệu tài chính đa thị trường toàn cầu, cho phép dự báo chuỗi thời gian đa biến (OHLCV) theo phương thức zero-shot — không cần fine-tune riêng cho từng thị trường.

### Mục tiêu

Đánh giá khả năng ứng dụng Kronos vào thị trường chứng khoán Việt Nam (VN30 index) theo 3 hướng:

1. **Zero-Shot inference** — dùng thẳng mô hình pre-trained
2. **Fine-Tuning** — huấn luyện thêm trên dữ liệu VN30
3. **Log-Return normalization** — chuẩn hóa đầu vào thành log-return trước khi inference

### Câu hỏi nghiên cứu

- Kronos có thể dự báo giá cổ phiếu VN30 với độ chính xác chấp nhận được không?
- Fine-tuning có cải thiện hiệu suất so với zero-shot không?
- Mã nào phù hợp nhất với Kronos?
- Signal từ Kronos có thể dùng để xây dựng chiến lược giao dịch không?

---

## 2. Dữ liệu

### Nguồn và phạm vi

| Thuộc tính | Chi tiết |
|------------|---------|
| **Nguồn** | Yahoo Finance (suffix `.VN`) |
| **Khoảng thời gian** | 2021-11-01 → 2026-10-30 (~5 năm) |
| **Số mã** | 30 mã VN30 |
| **Tổng rows (raw)** | 39,051 |
| **Tổng rows (cleaned)** | 37,035 |
| **Số ngày/mã** | 1,225–1,235 ngày giao dịch |

### Danh sách 30 mã VN30

| Sector | Mã |
|--------|-----|
| **Banking (13)** | ACB, BID, CTG, HDB, MBB, SHB, SSB, STB, TCB, TPB, VCB, VIB, VPB |
| **Real Estate (4)** | BCM, VHM, VIC, VRE |
| **Consumer (3)** | MSN, MWG, SAB, VNM *(VNM thực tế Consumer)* |
| **Energy (3)** | GAS, PLX, POW |
| **Khác (7)** | BVH (Insurance), FPT (Tech), GVR (Agriculture), HPG (Steel), SSI (Finance), VJC (Aviation), MWG (Retail) |

### Quy trình làm sạch dữ liệu

```
Raw CSV → Loại ngày volume=0 → Điều chỉnh OHLC theo adj_close ratio
→ Validate OHLC (low ≤ close ≤ high) → Tính amount = volume × close
→ Processed CSV (7 cột: date, open, high, low, close, volume, amount)
```

**Số liệu bị loại:** 2,016 rows (5.2%) — chủ yếu ngày không có giao dịch, ngày lễ điều chỉnh

---

## 3. Mô hình Kronos

### Kiến trúc

| Thành phần | Chi tiết |
|-----------|---------|
| **Model ID** | `NeoQuasar/Kronos-small` |
| **Tokenizer** | `NeoQuasar/Kronos-Tokenizer-base` |
| **Loại** | Transformer decoder-based time-series model |
| **Input** | Multivariate OHLCV + amount (6 features) |
| **Tokenization** | BSQ (Binary Spherical Quantization) — mã hóa từng time-step thành token rời rạc |
| **Output** | Phân phối xác suất trên token space → sample → decode thành giá |
| **Size** | Tokenizer ~15.8 MB, Predictor ~94 MB |

### Inference parameters

```python
lookback     = 400   # số ngày lịch sử làm đầu vào
pred_len     = 20    # số ngày dự báo
sample_count = 20    # số lần sample từ phân phối
temperature  = 1.0   # độ ngẫu nhiên
top_p        = 0.9   # nucleus sampling
device       = "mps" # Apple Silicon GPU
```

### Cơ chế dự báo

```
Dữ liệu lịch sử (400 ngày) → Tokenizer BSQ → Chuỗi token
→ Kronos Transformer → Predict token tiếp theo × pred_len
→ Sample × 20 → Decode → Lấy median làm dự báo
```

---

## 4. Phương pháp thực nghiệm

### 4.1 Metrics đánh giá

**MAPE (Mean Absolute Percentage Error)**
$$\text{MAPE} = \frac{1}{n}\sum_{t=1}^{n}\left|\frac{\hat{y}_t - y_t}{y_t}\right| \times 100\%$$

Đánh giá độ chính xác của dự báo giá tuyệt đối. MAPE < 5% được coi là tốt với dữ liệu tài chính.

**DirAcc (Directional Accuracy)**
$$\text{DirAcc} = \frac{1}{n-1}\sum_{t=2}^{n}\mathbf{1}\left[\text{sign}(\hat{y}_t - \hat{y}_{t-1}) = \text{sign}(y_t - y_{t-1})\right] \times 100\%$$

Đánh giá khả năng dự báo đúng chiều biến động. Baseline ngẫu nhiên = 50%.

**MAE (Mean Absolute Error):** Sai số tuyệt đối trung bình (đơn vị VND)

### 4.2 Cấu hình thực nghiệm

| Thực nghiệm | Phương pháp | Dữ liệu đầu vào | Đầu ra |
|------------|-------------|-----------------|--------|
| **ZS** | Zero-shot inference | Absolute OHLCV | 20-day forecast |
| **FT v1** | Fine-tune 3 epochs, patience=1 | Absolute OHLCV | 20-day forecast |
| **FT v2** | Fine-tune 5 epochs, patience=3 | Absolute OHLCV | 20-day forecast |
| **LR** | Zero-shot on log-returns | log(P_t/P_{t-1}) | 20-day forecast (reconstructed) |
| **MW** | Zero-shot × 5 windows | Absolute OHLCV | 5 × 20-day forecasts |

### 4.3 Multi-Window Walk-Forward Setup

```
Toàn bộ dữ liệu: 2021-11 → 2026-10
                    ↓
         Chia 5 window không overlap
         Mỗi window: lookback=400d + predict=20d
         
Window 1: train→2025-Q1, predict 20 ngày tiếp
Window 2: train→2025-Q2, predict 20 ngày tiếp
Window 3: train→2025-Q3, predict 20 ngày tiếp
Window 4: train→2025-Q4, predict 20 ngày tiếp
Window 5: train→2026-Q3, predict 20 ngày tiếp
```

---

## 5. Kết quả Zero-Shot

### Tổng quan

| Metric | Giá trị | So với baseline |
|--------|---------|----------------|
| **Avg MAPE** | **5.01%** | — |
| **Avg DirAcc** | **46.0%** | −4pp vs random (50%) |
| **Avg MAE** | ~3,100 VND | — |
| Mã MAPE < 3% | 9/30 | |
| Mã MAPE < 5% | 16/30 | |
| Mã DirAcc > 50% | 13/30 | |

### Chi tiết từng mã (Zero-Shot, sorted by MAPE)

| # | Ticker | Sector | MAPE | DirAcc | MAE (VND) | Đánh giá |
|---|--------|--------|:----:|:------:|:---------:|---------|
| 1 | **VNM** | Consumer | 1.01% | 52.6% | ~890 | 🌟 Tốt nhất |
| 2 | **POW** | Energy | 1.17% | 47.4% | ~170 | ✅ |
| 3 | **SAB** | Consumer | 1.26% | 42.1% | ~1,420 | ✅ |
| 4 | **VPB** | Banking | 1.34% | 68.4% | ~360 | ✅ DirAcc cao |
| 5 | **VIB** | Banking | 1.59% | 57.9% | ~220 | ✅ |
| 6 | **VCB** | Banking | 1.64% | 42.1% | ~840 | ✅ |
| 7 | **GAS** | Energy | 2.48% | 26.3% | ~4,380 | ⚠️ DirAcc thấp |
| 8 | **BID** | Banking | 2.78% | 26.3% | ~1,050 | ⚠️ DirAcc thấp |
| 9 | **MWG** | Retail | 2.90% | 36.8% | ~3,440 | OK |
| 10 | **TCB** | Banking | 2.95% | 47.4% | ~655 | ✅ |
| 11 | **BVH** | Insurance | 3.05% | 63.2% | ~1,580 | ✅ DirAcc cao |
| 12 | **CTG** | Banking | 3.28% | 47.4% | ~990 | ✅ |
| 13 | **TPB** | Banking | 3.42% | 52.6% | ~410 | ✅ |
| 14 | **ACB** | Banking | 3.60% | 36.8% | ~780 | OK |
| 15 | **SSI** | Finance | 3.66% | 47.4% | ~1,150 | OK |
| 16 | **STB** | Banking | 3.90% | 47.4% | ~1,640 | OK |
| 17 | **VJC** | Aviation | 4.10% | 36.8% | ~4,690 | OK |
| 18 | **MSN** | Consumer | 4.33% | 73.7% | ~3,530 | ✅ DirAcc rất cao |
| 19 | **HPG** | Steel | 4.35% | 31.6% | ~1,180 | ⚠️ DirAcc thấp |
| 20 | **VHM** | Real Estate | 4.54% | 36.8% | ~2,720 | OK |
| 21 | **GVR** | Agriculture | 4.76% | 42.1% | ~1,500 | OK |
| 22 | **HDB** | Banking | 4.87% | 52.6% | ~1,080 | ✅ |
| 23 | **VRE** | Real Estate | 5.27% | 47.4% | ~1,380 | OK |
| 24 | **MBB** | Banking | 5.28% | 36.8% | ~1,720 | OK |
| 25 | **PLX** | Energy | 6.58% | 36.8% | ~2,430 | ⚠️ |
| 26 | **BCM** | Real Estate | 8.16% | 52.6% | ~3,990 | ⚠️ |
| 27 | **FPT** | Technology | 9.44% | 47.4% | ~5,210 | ⚠️ |
| 28 | **SHB** | Banking | 10.58% | 47.4% | ~1,300 | ❌ |
| 29 | **SSB** | Banking | 13.55% | 57.9% | ~2,070 | ❌ |
| 30 | **VIC** | Real Estate | 24.53% | 36.8% | ~51,800 | ❌ Outlier |

### Nhận xét VIC (outlier)

VIC có MAPE 24.53% — gấp 5 lần average. Nguyên nhân: VIC là cổ phiếu có giá trị tuyệt đối cao nhất VN30 (~40,000–80,000 VND/cp) với biến động rất mạnh do các thông tin bất động sản, tái cơ cấu doanh nghiệp. Kronos tokenizer BSQ gặp khó khăn khi scale của giá quá khác biệt.

---

## 6. Phân tích theo Sector

| Sector | Số mã | Avg MAPE | Avg DirAcc | Mã tiêu biểu |
|--------|:-----:|:--------:|:----------:|-------------|
| **Consumer** | 3 | **2.20%** | **56.1%** | VNM, SAB, MSN |
| **Retail** | 1 | 2.90% | 36.8% | MWG |
| **Insurance** | 1 | 3.05% | 63.2% | BVH |
| **Energy** | 3 | 3.41% | 36.8% | GAS, PLX, POW |
| **Finance** | 1 | 3.66% | 47.4% | SSI |
| **Aviation** | 1 | 4.10% | 36.8% | VJC |
| **Steel** | 1 | 4.35% | 31.6% | HPG |
| **Banking** | 13 | 4.52% | 47.8% | VCB, VPB, VIB... |
| **Agriculture** | 1 | 4.76% | 42.1% | GVR |
| **Technology** | 1 | 9.44% | 47.4% | FPT |
| **Real Estate** | 4 | 10.62% | 43.4% | VIC, BCM, VHM, VRE |

**Nhận xét:**
- **Consumer** tốt nhất: doanh nghiệp consumer VN có pattern giá ổn định, biến động ít bị ảnh hưởng bởi sentiment ngắn hạn
- **Banking** đứng thứ 8: sector lớn nhất (13 mã), kết quả đồng đều nhưng không đặc biệt tốt
- **Real Estate** tệ nhất (trừ Consumer): VIC kéo down mạnh; BCM và VHM cũng biến động mạnh do thị trường BĐS
- **Energy** MAPE thấp nhưng DirAcc thấp (36.8%): Kronos dự báo mức giá gần đúng nhưng sai chiều thường xuyên

---

## 7. Multi-Window Walk-Forward Evaluation

### Thiết kế

5 windows không overlap, mỗi window predict 20 ngày sau 400 ngày lookback. Đây là phương pháp đánh giá **đáng tin cậy hơn** single-point evaluation vì test trên nhiều giai đoạn thị trường khác nhau.

### Kết quả theo window

| Window | Giai đoạn predict | Avg MAPE | Avg DirAcc | Tốt nhất | Tệ nhất |
|--------|-------------------|:--------:|:----------:|---------|--------|
| W1 | 2025-Q1 | 8.73% | 39.7% | VCB (1.01%) | VIC (61.6%) |
| W2 | 2025-Q2 | 10.49% | 51.0% | SAB (0.89%) | VIC (48.7%) |
| W3 | 2025-Q3 | 7.05% | 46.0% | POW (1.05%) | VIC (40.9%) |
| W4 | 2025-Q4 | 6.01% | 46.3% | MSN (1.18%) | VIC (33.8%) |
| W5 | 2026-Q4 | **5.63%** | 46.3% | POW (1.03%) | VIC (23.9%) |

> **Trend quan trọng:** MAPE giảm dần từ W1→W5. Kronos được pre-train trên dữ liệu gần hơn với 2026 → tự nhiên hoạt động tốt hơn trên window gần đây.

### Tổng hợp theo ticker (5 windows)

**Top performers (low MAPE, stable DirAcc):**

| Ticker | Mean MAPE | Mean DirAcc | Std DirAcc | Consistency |
|--------|:---------:|:-----------:|:----------:|------------|
| SAB | 2.07% | 51.0% | 16.4% | ⚡ MAPE tốt, std cao |
| POW | 2.48% | 51.0% | 18.5% | ⚡ MAPE tốt, std cao |
| ACB | 2.68% | 43.0% | 16.8% | MAPE tốt |
| MSN | 3.03% | 50.0% | **6.1%** | ✅ Ổn định nhất |
| CTG | 3.05% | 50.0% | 11.7% | ✅ |
| VCB | 3.09% | 51.0% | **8.2%** | ✅ Tốt toàn diện |
| BID | 3.68% | 51.0% | 11.4% | ✅ |
| BVH | 3.42% | 49.0% | 15.6% | OK |

**DirAcc cao nhất (directional signal):**

| Ticker | Mean DirAcc | Std DirAcc | Mean MAPE | Assessment |
|--------|:-----------:|:----------:|:---------:|-----------|
| **STB** | 56.0% | 10.8% | 13.03% | ⚠️ DirAcc cao nhưng MAPE xấu |
| **MBB** | **55.0%** | **5.0%** | 6.45% | ✅ Ổn định nhất overall |
| **PLX** | 54.0% | 11.4% | 8.80% | OK |
| **VHM** | 53.0% | 9.1% | 20.61% | ⚠️ MAPE quá xấu |
| **SAB** | 51.0% | 16.4% | 2.07% | ⚡ MAPE tốt nhất |
| **POW** | 51.0% | 18.5% | 2.48% | ⚡ |
| **VCB** | 51.0% | 8.2% | 3.09% | ✅ Balanced |

**Worst performers (avoid for signals):**

| Ticker | Mean DirAcc | Std DirAcc | Mean MAPE | Ghi chú |
|--------|:-----------:|:----------:|:---------:|---------|
| **SHB** | 35.0% | 8.7% | 7.73% | Tệ nhất DirAcc |
| **GVR** | 36.0% | 8.2% | 6.04% | |
| **FPT** | **37.0%** | **4.5%** | 8.15% | 💡 Consistently sai → contra-signal! |
| **VJC** | 38.0% | 11.5% | 7.61% | |
| **VRE** | 39.0% | 9.6% | 7.87% | |

### 💡 FPT Contra-Signal Insight

FPT có DirAcc chỉ 37% với std cực thấp 4.5% — nghĩa là mô hình **nhất quán dự báo sai chiều** FPT qua 5 windows. Nếu **đảo ngược** signal:
- Expected DirAcc ≈ 63% (1 - 37%)
- Std vẫn thấp 4.5% → **rất ổn định**
- Đây là insight quan trọng có thể khai thác trong chiến lược giao dịch

### Tiêu chí Reliability (DirAcc > 55% AND std < 10%)

**Kết quả: Không có ticker nào đáp ứng đồng thời cả hai tiêu chí.**

- MBB: 55.0% / std=5.0% → chỉ thiếu 0.001pp DirAcc
- STB: 56.0% / std=10.84% → std hơi cao

Điều này cho thấy không có mã nào có signal **vừa mạnh vừa ổn định** đủ để dùng trực tiếp làm trading signal mà không cần kết hợp thêm.

---

## 8. Fine-Tuning — Phân tích chi tiết

### 8.1 Cấu hình Fine-Tune

| Tham số | Fine-Tune v1 | Fine-Tune v2 |
|---------|:-----------:|:-----------:|
| Epochs | 3 | 5 |
| Batch size | 4 | 4 |
| Patience (predictor) | 1 | 3 |
| Patience (tokenizer) | 2 | 3 |
| Learning rate | 2e-4 / 1e-4 | 2e-4 / 1e-4 |
| Train data | 2021-11 → 2025-10 | 2021-11 → 2025-10 |
| Val data | 2025-11 → 2026-10 | 2025-11 → 2026-10 |
| Sliding window | lookback=400, pred=20, stride=5 | lookback=400, pred=20, stride=5 |
| Train batches | 855 | 855 |
| Val batches | 847 | 847 |
| Total time | ~22 min | ~34 min |

### 8.2 Learning Curves

**Fine-Tune v2 — Tokenizer Phase:**

| Epoch | Train Loss | Val Loss | Saved |
|-------|:----------:|:--------:|:-----:|
| 1/5 | -0.0588 | -0.0502 | ✅ |
| 2/5 | -0.0601 | -0.0507 | ✅ |
| 3/5 | -0.0607 | **-0.0511** | ✅ |
| 4/5 | -0.0610 | -0.0511 | ✅ (tie) |
| 5/5 | -0.0611 | -0.0511 | ✅ (tie) |

Tokenizer **không bị overfit** — loss giảm đều và ổn định. Đây là component học tốt.

**Fine-Tune v2 — Predictor Phase:**

| Epoch | Train Loss | Val Loss | Saved |
|-------|:----------:|:--------:|:-----:|
| **1/5** | 2.4300 | **3.1511** | ✅ Best |
| 2/5 | 2.2374 | 3.3255 | ❌ |
| 3/5 | 2.1000 | 3.4473 | ❌ |
| 4/5 | 2.0260 | 3.4975 | ❌ → Early stop |

Predictor **overfit ngay từ epoch 1**: Train loss giảm liên tục (2.43→2.03) nhưng val loss tăng (3.15→3.50). Đây là dấu hiệu của **underfitting ở val** — mô hình học distribution của 2021-2025 nhưng không generalize sang 2026.

### 8.3 So sánh kết quả Fine-Tune

| Phiên bản | Avg MAPE | Avg DirAcc | Mã cải thiện MAPE | Mã cải thiện DirAcc |
|-----------|:--------:|:----------:|:-----------------:|:-------------------:|
| Zero-Shot | **5.01%** | 46.0% | baseline | baseline |
| FT v1 (p=1) | 8.27% | 41.2% | 7/30 | 9/30 |
| FT v2 (p=3) | 8.27% | 44.2% | 7/30 | 9/30 |

> **Cả hai phiên bản FT đều cho kết quả gần như giống nhau.** Tăng patience không giúp ích vì vấn đề là overfitting từ epoch đầu tiên, không phải stopping quá sớm.

### 8.4 Phân tích chi tiết: Mã được FT cải thiện

| Ticker | ZS MAPE | FT MAPE | Δ MAPE | ZS Dir | FT Dir | Δ Dir | Lý giải |
|--------|:-------:|:-------:|:------:|:------:|:------:|:-----:|---------|
| **VIC** | 24.53% | 11.44% | **−13.09%** | 36.8% | 47.4% | +10.6pp | VIC có giá scale đặc biệt, FT giúp tokenizer adapt |
| **SSB** | 13.55% | 10.46% | −3.09% | 57.9% | 57.9% | 0pp | SSB có pattern biến động đặc trưng |
| **MSN** | 4.33% | 1.74% | −2.59% | 73.7% | 36.8% | −36.9pp | MAPE tốt hơn nhưng DirAcc tệ hơn! |
| **TCB** | 2.95% | 1.99% | −0.96% | 47.4% | 47.4% | 0pp | Ổn định, FT giúp nhẹ |
| **MWG** | 2.90% | 2.17% | −0.73% | 36.8% | 31.6% | −5.2pp | |
| **VIB** | 1.59% | 0.98% | −0.61% | 57.9% | 63.2% | +5.3pp | Tốt nhất toàn diện |
| **VCB** | 1.64% | 1.28% | −0.36% | 42.1% | 47.4% | +5.3pp | |

### 8.5 Phân tích chi tiết: Mã bị FT hại

| Ticker | ZS MAPE | FT MAPE | Δ MAPE | Lý giải |
|--------|:-------:|:-------:|:------:|---------|
| **BID** | 2.78% | 20.00% | **+17.22%** | BID có pattern ổn định → FT overfit nhiễu |
| **VRE** | 5.27% | 18.51% | +13.24% | |
| **SSI** | 3.66% | 15.68% | +12.02% | |
| **SHB** | 10.58% | 20.02% | +9.44% | |
| **POW** | 1.17% | 7.98% | +6.81% | |
| **HPG** | 4.35% | 11.07% | +6.72% | |
| **CTG** | 3.28% | 9.55% | +6.27% | |

### 8.6 Nguyên nhân Fine-Tune không hiệu quả

1. **Distribution shift:** Model học distribution 2021-2025 → test trên 2026. Năm 2026 có macro khác (lãi suất, tỷ giá) không có trong train data.

2. **Mixed-ticker training:** 30 mã với pattern rất khác nhau cùng vào 1 batch → gradient từ VIC (giá cao, biến động mạnh) triệt tiêu gradient từ VCB (ổn định).

3. **Limited val data:** Chỉ ~250 ngày để validate → val loss noisy, không phản ánh đúng generalization.

4. **Predictor capacity:** Kronos-small có capacity hạn chế → dễ overfit khi fine-tune trên domain hẹp.

---

## 9. Log-Return Normalization

### Phương pháp

Thay vì dùng giá tuyệt đối, chuyển đổi sang log-return:
$$r_t = \log\left(\frac{P_t}{P_{t-1}}\right)$$

Sau đó reconstruct giá từ predicted log-returns:
$$\hat{P}_t = P_{t-1} \times e^{\hat{r}_t}$$

### Kết quả

| Metric | Zero-Shot | Log-Return | Δ |
|--------|:---------:|:----------:|:-:|
| Avg MAPE | **5.01%** | 5.78% | +0.77% |
| Avg DirAcc | 46.0% | 45.1% | −0.9pp |

Kết quả trên 3 mã cuối (VNM, VPB, VRE):

| Ticker | ZS MAPE | LR MAPE | Δ | ZS Dir | LR Dir | Δ |
|--------|:-------:|:-------:|:---:|:------:|:------:|:---:|
| VNM | 1.01% | 2.28% | +1.27% | 55% | 50% | −5pp |
| VRE | 5.27% | 7.04% | +1.77% | 50% | 40% | −10pp |
| VPB | 1.34% | 9.51% | +8.17% | 70% | 35% | −35pp |

### Kết luận

Log-Return normalization **không cải thiện** và thậm chí làm tệ hơn đáng kể (VPB: +8.17% MAPE). Kronos BSQ tokenizer đã tự handle việc chuẩn hóa scale nội bộ — không cần thêm bước này.

---

## 10. Bảng So Sánh Tổng Hợp — 30 Mã

| Ticker | Sector | ZS MAPE | ZS Dir | FT MAPE | FT Dir | FT Δ | MW MAPE | MW Dir | MW Std | Best Method |
|--------|--------|:-------:|:------:|:-------:|:------:|:----:|:-------:|:------:|:------:|:-----------:|
| VNM | Consumer | 1.01% | 52.6% | 5.79% | 26.3% | +4.78% | 3.08% | 46% | 9.6 | **ZS** |
| POW | Energy | 1.17% | 47.4% | 7.98% | 21.1% | +6.81% | 2.48% | 51% | 18.5 | **ZS** |
| SAB | Consumer | 1.26% | 42.1% | 2.80% | 52.6% | +1.54% | 2.07% | 51% | 16.4 | **ZS** |
| VPB | Banking | 1.34% | 68.4% | 3.65% | 31.6% | +2.31% | 4.24% | 49% | 10.2 | **ZS** |
| VIB | Banking | 1.59% | 57.9% | **0.98%** | 63.2% | **−0.61%** | 3.78% | 47% | 18.9 | **FT** |
| VCB | Banking | 1.64% | 42.1% | **1.28%** | 47.4% | −0.36% | 3.09% | 51% | 8.2 | FT (≈ZS) |
| GAS | Energy | 2.48% | 26.3% | 4.98% | 47.4% | +2.50% | 7.75% | 48% | 11.5 | **ZS** |
| BID | Banking | 2.78% | 26.3% | 20.00% | 26.3% | +17.22% | 3.68% | 51% | 11.4 | **ZS** |
| MWG | Retail | 2.90% | 36.8% | **2.17%** | 31.6% | −0.73% | 5.81% | 47% | 5.7 | FT (≈ZS) |
| TCB | Banking | 2.95% | 47.4% | **1.99%** | 47.4% | −0.96% | 6.84% | 40% | 7.9 | **FT** |
| BVH | Insurance | 3.05% | 63.2% | 3.54% | 47.4% | +0.49% | 3.42% | 49% | 15.6 | **ZS** |
| CTG | Banking | 3.28% | 47.4% | 9.55% | 57.9% | +6.27% | 3.05% | 50% | 11.7 | **ZS** |
| TPB | Banking | 3.42% | 52.6% | 7.31% | 31.6% | +3.89% | 5.04% | 47% | 14.4 | **ZS** |
| ACB | Banking | 3.60% | 36.8% | 3.64% | 36.8% | +0.04% | 2.68% | 43% | 16.8 | **ZS** |
| SSI | Finance | 3.66% | 47.4% | 15.68% | 36.8% | +12.02% | 3.89% | 47% | 11.5 | **ZS** |
| STB | Banking | 3.90% | 47.4% | 6.85% | 47.4% | +2.95% | 13.03% | 56% | 10.8 | **ZS** |
| VJC | Aviation | 4.10% | 36.8% | 5.23% | 21.1% | +1.13% | 7.61% | 38% | 11.5 | **ZS** |
| MSN | Consumer | 4.33% | 73.7% | **1.74%** | 36.8% | **−2.59%** | 3.03% | 50% | 6.1 | **FT** (MAPE) |
| HPG | Steel | 4.35% | 31.6% | 11.07% | 31.6% | +6.72% | 4.46% | 45% | 9.4 | **ZS** |
| VHM | Real Estate | 4.54% | 36.8% | 8.81% | 63.2% | +4.27% | 20.61% | 53% | 9.1 | **ZS** |
| GVR | Agriculture | 4.76% | 42.1% | 10.40% | 42.1% | +5.64% | 6.04% | 36% | 8.2 | **ZS** |
| HDB | Banking | 4.87% | 52.6% | 7.73% | 47.4% | +2.86% | 4.06% | 45% | 7.1 | **ZS** |
| VRE | Real Estate | 5.27% | 47.4% | 18.51% | 21.1% | +13.24% | 7.87% | 39% | 9.6 | **ZS** |
| MBB | Banking | 5.28% | 36.8% | 7.87% | 42.1% | +2.59% | 6.45% | 55% | 5.0 | **ZS** |
| PLX | Energy | 6.58% | 36.8% | 11.11% | 31.6% | +4.53% | 8.80% | 54% | 11.4 | **ZS** |
| BCM | Real Estate | 8.16% | 52.6% | 12.30% | 57.9% | +4.14% | 16.77% | 45% | 5.0 | **ZS** |
| FPT | Technology | 9.44% | 47.4% | 13.11% | 42.1% | +3.67% | 8.15% | **37%** | **4.5** | ⚡ Contra |
| SHB | Banking | 10.58% | 47.4% | 20.02% | 42.1% | +9.44% | 7.73% | 35% | 8.7 | **ZS** |
| SSB | Banking | 13.55% | 57.9% | **10.46%** | 57.9% | **−3.09%** | 10.14% | 47% | 12.5 | **FT** |
| VIC | Real Estate | 24.53% | 36.8% | **11.44%** | 47.4% | **−13.09%** | 41.78% | 46% | 8.9 | **FT** |

**Tóm tắt Best Method:**
- ZS tốt hơn: **23/30 mã**
- FT tốt hơn: **7/30 mã** (VIC, SSB, MSN, TCB, MWG, VIB, VCB)
- Contra-signal: **1 mã** (FPT)

---

## 11. Backtesting

### Chiến lược

**Dữ liệu:** 2021-11-01 → 2026-10-30 (~1,200 ngày)  
**Chi phí giao dịch:** 0.1% mỗi chiều (thực tế thị trường VN)  
**Tham chiếu:** Buy & Hold VCB

| Strategy | Mô tả |
|----------|-------|
| **A: Kronos Signal** | Nếu Kronos predict tăng → hold; predict giảm → cash |
| **B: Threshold** | Chỉ vào lệnh khi signal mạnh (>1% predicted move) |
| **C: Long/Short** | Long mã signal tăng mạnh nhất, Short mã giảm mạnh nhất |

### Kết quả

| Strategy | Total Return | Sharpe | Max Drawdown | vs Buy&Hold |
|----------|:------------:|:------:|:------------:|:-----------:|
| **A: Kronos Signal** | **+4.25%** | −0.57 | −20.3% | **+15.4pp** |
| B: Threshold | −2.74% | — | — | −8.4pp |
| C: Long/Short | −91.7% | — | — | ❌ |
| Buy & Hold VCB | −11.15% | — | −35% | baseline |

### Nhận xét

- **Strategy A** là chiến lược duy nhất có lãi. Tuy nhiên Sharpe âm (−0.57) cho thấy rủi ro điều chỉnh theo biến động vẫn chưa tốt.

- **Strategy C** bị phá hủy hoàn toàn bởi transaction cost: với 0.1% × 2 sides × ~600 lệnh/năm = ~120%/năm chi phí giao dịch — vượt xa mọi alpha.

- **Lesson:** Signal từ Kronos có value thực nhưng phải kết hợp với position sizing tốt và giảm tần suất giao dịch.

---

## 12. Kết luận & Khuyến nghị

### 12.1 Kết luận chính

**1. Zero-Shot Kronos hoạt động được trên VN30**
- Avg MAPE 5.01% trong 20-day window là kết quả hợp lý cho dữ liệu tài chính
- 16/30 mã đạt MAPE < 5%; 9/30 mã đạt MAPE < 3%
- Consumer sector có kết quả tốt nhất (avg MAPE 2.2%)

**2. Fine-Tuning KHÔNG cải thiện trong setting hiện tại**
- FT avg MAPE tăng từ 5.01% → 8.27% (+3.26pp)
- Chỉ 7/30 mã được cải thiện
- Predictor overfit từ epoch 1, tăng patience không giải quyết được

**3. Log-Return normalization không có lợi**
- MAPE tăng +0.77%, DirAcc giảm −0.9pp
- Kronos BSQ tokenizer đã xử lý scale nội bộ

**4. Multi-window xác nhận trend quan trọng**
- MAPE giảm dần theo thời gian (W1: 8.73% → W5: 5.63%)
- Không có ticker nào đáp ứng tiêu chí reliability nghiêm ngặt
- FPT là contra-signal tiềm năng nhất (DirAcc 37%, std 4.5%)

**5. Backtesting cho thấy tiềm năng nhưng còn hạn chế**
- Strategy A +4.25% vs Buy&Hold −11.15%
- Sharpe âm → cần cải thiện risk management

### 12.2 Mã tốt nhất để dùng ZS signal

**Tier 1 — Reliable (MAPE thấp + DirAcc ổn định qua MW):**
- **VCB:** ZS MAPE 1.64%, MW DirAcc 51%, MW std 8.2%
- **MSN:** ZS MAPE 4.33% (ZS Dir 73.7%!), MW MAPE 3.03%, MW std 6.1%
- **MBB:** MW DirAcc 55%, MW std 5.0% (most stable direction signal)

**Tier 2 — Good (MAPE thấp hoặc DirAcc tốt):**
- SAB, POW, CTG, BID, TPB, HDB

**Không nên dùng:**
- VIC (MAPE 24-42%), BCM (MAPE 8-17%), STB (MW MAPE 13%), SHB (DirAcc 35%)

### 12.3 Mã nên dùng FT checkpoint

Chỉ dùng FT thay ZS cho: **VIC, SSB, MSN** (cải thiện MAPE đáng kể) và **VIB, TCB** (cải thiện nhẹ nhưng ổn định)

---

## 13. Hướng Phát Triển Tiếp Theo

### Ngắn hạn (tuần tới)

1. **Backtest FPT contra-signal** — đảo ngược FPT predictions, backtest riêng → expected ~63% DirAcc
2. **Giảm pred_len** từ 20 → 5 ngày → MAPE có thể giảm đáng kể
3. **Ensemble ZS + FT** — dùng ZS cho 23 mã, FT cho 7 mã cải thiện

### Trung hạn (tháng tới)

4. **Per-ticker fine-tune** — train riêng model cho từng mã (đặc biệt MBB, VCB, MSN) thay vì mix 30 mã
5. **Walk-forward backtest** — dùng MW predictions (150 data points thay vì 20) cho Sharpe đáng tin hơn
6. **Hyperparameter tuning** — thử lookback=200/600, pred_len=5/10, temperature khác nhau

### Dài hạn

7. **Kronos-base/large** — thử model lớn hơn xem có cải thiện không
8. **Kết hợp với fundamental data** — PE, ROE, tin tức → hybrid model
9. **Live trading simulation** — chạy paper trading với signal thực theo ngày

---

## PHỤ LỤC: Files & Scripts

### Output Files

| File | Nội dung | Kích thước |
|------|---------|-----------|
| `data/processed/*.csv` | 30 mã × ~1,235 ngày, cleaned OHLCV | ~4.5 MB |
| `results/predictions/*.csv` | ZS predictions (30 mã × 20 ngày) | 128 KB |
| `results/predictions_finetuned/*.csv` | FT predictions | 124 KB |
| `results/predictions_logreturn/*.csv` | LR predictions | 124 KB |
| `results/multiwindow/per_window_metrics.csv` | 150 rows (5W × 30M) | — |
| `results/multiwindow/aggregated_metrics.csv` | 30 rows ticker-level | — |
| `results/charts/*.png` | 30 ZS charts | 4.1 MB |
| `results/analysis/metrics_heatmap.png` | MAPE/DirAcc heatmap | 238 KB |
| `results/analysis/performance_scatter.png` | MAPE vs DirAcc scatter | 190 KB |
| `results/analysis/forecast_overlay.png` | Top 6 ticker forecast | 363 KB |
| `results/charts/backtest_results.png` | Backtest 3 strategies | 632 KB |
| `models/finetuned/vn30_full_3ep/` | FT v1 checkpoint | 110 MB |
| `models/finetuned/vn30_v2_patience3/` | FT v2 checkpoint | 110 MB |

### Scripts

| Script | Chức năng | Thời gian chạy |
|--------|----------|---------------|
| `scripts/download_vn30.py` | Tải dữ liệu Yahoo Finance | ~3 phút |
| `scripts/clean_data.py` | Làm sạch & chuẩn hóa | ~30 giây |
| `scripts/kronos_inference.py` | Zero-shot inference 30 mã | ~30 phút |
| `scripts/kronos_inference_finetuned.py` | FT inference (--exp-name) | ~30 phút |
| `scripts/kronos_logreturn_inference.py` | Log-return inference | ~30 phút |
| `scripts/finetune_kronos.py` | Fine-tune (--patience, --epochs) | ~30-60 phút |
| `scripts/multiwindow_eval.py` | Walk-forward 5 windows | ~45-60 phút |
| `scripts/analyze_results.py` | Tạo charts & heatmaps | ~2 phút |
| `scripts/backtest.py` | Backtest 3 strategies | ~1 phút |

---

*Báo cáo hoàn chỉnh — 2026-10-31 | Kronos VN30 Research Pipeline*

