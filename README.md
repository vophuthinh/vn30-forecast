# Kronos VN30 — Dự báo giá chứng khoán Việt Nam

Ứng dụng mô hình [**Kronos**](https://github.com/shiyu-coder/Kronos) (Tsinghua University, AAAI) cho thị trường chứng khoán Việt Nam (VN30).

## 📁 Cấu trúc dự án

```
kronos-vn30/
├── data/
│   ├── raw/            # OHLCV thô từ Yahoo Finance (30 file CSV)
│   └── processed/      # Dữ liệu đã làm sạch + adj_close + amount
├── notebooks/          # Jupyter notebooks phân tích & thử nghiệm
├── src/                # Source code (loader, predictor, evaluator)
├── models/             # Model checkpoints (Kronos weights)
├── results/            # Kết quả dự báo, metrics, biểu đồ
└── scripts/
    ├── download_vn30.py   # Thu thập dữ liệu từ Yahoo Finance
    └── preprocess.py      # Làm sạch & chuẩn hóa
```

## ⚡ Quickstart

```bash
# 1. Clone Kronos
git clone https://github.com/shiyu-coder/Kronos

# 2. Cài dependencies
pip install -r requirements.txt

# 3. Cập nhật dữ liệu mới nhất
python scripts/download_vn30.py

# 4. Tiền xử lý
python scripts/preprocess.py
```

## 📊 Dữ liệu

| Thông tin | Giá trị |
|-----------|---------|
| Nguồn | Yahoo Finance (`.VN`) |
| Số mã | 30 (toàn bộ VN30) |
| Thời gian | 2021-11-01 → nay (~5 năm) |
| Tần suất | Daily (1D) |
| Columns | `date, open, high, low, close, volume, amount` |

### Danh sách VN30
`ACB · BCM · BID · BVH · CTG · FPT · GAS · GVR · HDB · HPG`
`MBB · MSN · MWG · PLX · POW · SAB · SHB · SSB · SSI · STB`
`TCB · TPB · VCB · VHM · VIB · VIC · VJC · VNM · VPB · VRE`

## 🤖 Mô hình Kronos

| Phiên bản | Tham số | HuggingFace |
|-----------|---------|-------------|
| Kronos-mini | ~4.1M | `NeoQuasar/Kronos-mini` |
| Kronos-small | ~24.7M | `NeoQuasar/Kronos-small` |
| Kronos-base | ~102.3M | `NeoQuasar/Kronos-base` |

## 🗺️ Lộ trình

- [x] Thu thập dữ liệu VN30 (30/30 mã)
- [ ] Làm sạch & chuẩn hóa dữ liệu
- [ ] Zero-shot inference với Kronos-small
- [ ] Đánh giá (MAE, MAPE, Direction Accuracy)
- [ ] Fine-tune trên dữ liệu VN
- [ ] Backtesting chiến lược giao dịch
