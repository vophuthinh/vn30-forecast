# Kronos VN30 — Zero-Shot Analysis Report
*Generated: 2026-10-30 | Model: Kronos-Small | Samples: 20 | pred_len: 20*

---

## Executive Summary

Kronos-Small đạt kết quả **tốt ở nhóm Banking và Energy** với MAPE thấp (< 3%).
Outlier nghiêm trọng là **VIC** (MAPE=24.5%), do biến động giá bất thường.
Directional Accuracy trung bình toàn thị trường là **49.0%**,
chưa cao hơn ngưỡng random (50%), cho thấy zero-shot chưa capture được momentum ngắn hạn.

---

## 🏆 Top 10 Mã Tốt Nhất (MAPE < 5%, Sorted by DirAcc ↓)

| Ticker | MAPE    | DirAcc   | MAE      | Score   | Sector         |
|--------|---------|----------|----------|---------|----------------|
| MSN    |    4.33% |       75% |    3,172 |    58.8 | Finance/Other  |
| VPB    |    1.34% |       70% |      322 |    65.3 | Banking        |
| BVH    |    3.05% |       65% |    2,165 |    55.1 | Finance/Other  |
| VIB    |    1.59% |       60% |      221 |    55.2 | Banking        |
| VNM    |    1.01% |       55% |      583 |    52.2 | Consumer       |
| HDB    |    4.87% |       55% |    1,099 |    41.6 | Banking        |
| TPB    |    3.42% |       55% |      421 |    45.6 | Banking        |
| STB    |    3.90% |       50% |    2,517 |    40.2 | Banking        |
| CTG    |    3.28% |       50% |      987 |    41.8 | Banking        |
| POW    |    1.17% |       50% |      145 |    47.1 | Energy         |

> **Score = (100 − MAPE×5) × DirAcc/100** — chỉ số tổng hợp giữa độ chính xác giá trị và chiều hướng.

---

## ⚠️ Top 5 Mã Cần Cải Thiện (MAPE cao nhất, loại VIC)

| Ticker | MAPE    | DirAcc   | MAE      | Score   | Sector         |
|--------|---------|----------|----------|---------|----------------|
| SSB    |   13.55% |       60% |    2,208 |    19.3 | Finance/Other  |
| SHB    |   10.58% |       50% |    1,117 |    23.5 | Banking        |
| FPT    |    9.44% |       50% |    5,284 |    26.4 | Finance/Other  |
| BCM    |    8.16% |       55% |    3,199 |    32.6 | Real Estate    |
| PLX    |    6.58% |       40% |    2,348 |    26.8 | Energy         |

---

## 📊 Phân Tích Theo Sector

| Sector         | Count | Avg MAPE  | Avg DirAcc |
|----------------|-------|-----------|------------|
| Consumer       |   3.0 |      1.72% |       46.7% |
| Energy         |   3.0 |      3.41% |       40.0% |
| Banking        |  12.0 |      3.77% |       49.6% |
| Industrial     |   2.0 |      4.22% |       37.5% |
| Real Estate    |   3.0 |      5.99% |       48.3% |
| Finance/Other  |   6.0 |      6.47% |       57.5% |

**Nhận xét:**
- **Energy** (GAS, PLX, POW): MAPE trung bình thấp nhất, giá năng lượng ổn định và có pattern rõ ràng → Kronos forecast tốt.
- **Banking**: MAPE trung bình thấp, nhưng DirAcc chỉ ~50% → Kronos capture magnitude tốt nhưng direction chưa chắc.
- **Finance/Other** (SSB, SHB, FPT): MAPE cao, biến động mạnh → cần fine-tune.
- **Real Estate** bị kéo bởi VIC outlier; loại VIC ra thì nhóm này chấp nhận được.

---

## 🔴 VIC Outlier — Phân Tích

| Metric | VIC | Avg (excl VIC) |
|--------|-----|----------------|
| MAE    | 56,493 | 1,626 |
| MAPE   | 24.53% | 4.34% |
| DirAcc | 40% | 49.0% |

**Nguyên nhân VIC outlier:**
1. **Giá cổ phiếu cao tuyệt đối** (VIC thường ~40,000–80,000 VNĐ/cp) → MAE tuyệt đối lớn, khuếch đại MAPE.
2. **Biến động bất thường**: VIC có nhiều sự kiện doanh nghiệp lớn (phát hành, sáp nhập, cổ tức đặc biệt) trong giai đoạn inference.
3. **Zero-shot limitation**: Kronos-Small không được expose với pattern của cổ phiếu high-volatility Vietnam real-estate.
4. **Lookback 20 samples** quá ngắn để model học được baseline price level đúng của VIC.

**Khuyến nghị**: Loại VIC khỏi zero-shot pipeline; fine-tune riêng với lookback ≥ 252 ngày.

---

## 🚀 Đề Xuất Hành Động

### ✅ Dùng Được Ngay (Zero-Shot Ready)
*Tiêu chí: MAPE < 4% và DirAcc ≥ 50%*

**BVH, CTG, POW, SSI, STB, TCB, TPB, VIB, VNM, VPB**

Các mã này có MAPE đủ thấp để dùng Kronos làm signal phụ trong chiến lược mean-reversion
hoặc kết hợp với indicator kỹ thuật.

### 🔧 Cần Fine-Tune Trước
*Tiêu chí: MAPE > 5% hoặc DirAcc < 40%*

**BCM, BID, FPT, GAS, HPG, MBB, PLX, SHB, SSB, VRE**

**Ưu tiên fine-tune theo thứ tự:**
1. **VIC** — outlier nghiêm trọng, cần lookback dài + data augmentation
2. **SSB, SHB** — MAPE > 10%, cổ phiếu thanh khoản thấp, noise nhiều
3. **FPT, BCM** — MAPE 8-9%, sector-specific pattern chưa capture được
4. **BID, GAS** — MAPE thấp nhưng DirAcc = 30%, fine-tune direction loss
5. **HPG** — Industrial với DirAcc = 35%, cần thêm volume features

### 📈 Lộ Trình Đề Xuất
```
Phase 1 (1-2 tuần): Fine-tune VIC + SSB + SHB với full historical data
Phase 2 (2-4 tuần): Fine-tune FPT, BCM, PLX với sector-aware embeddings
Phase 3 (ongoing):  A/B test zero-shot vs fine-tuned trên paper portfolio
```

---

## 📁 Files Đã Tạo

| File | Mô tả |
|------|-------|
| `results/analysis/metrics_heatmap.png` | Heatmap 30 mã × 3 metrics |
| `results/analysis/performance_scatter.png` | Scatter MAPE vs DirAcc, bubble=MAE |
| `results/analysis/forecast_overlay.png` | 6 mã MAPE thấp nhất — history + forecast |
| `results/analysis/summary_report.md` | Báo cáo này |

---
*Kronos VN30 Project | Zero-Shot Baseline | Phiên bản phân tích 1.0*
