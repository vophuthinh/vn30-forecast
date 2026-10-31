# VN30 — Log-Return vs Zero-Shot: Comparison Report

> Generated: 2026-10-31 00:20

## Summary

| Metric | Zero-Shot | Log-Return | Δ |
|--------|-----------|------------|---|
| Avg MAPE | 2.54% | 6.28% | +3.74% |
| Avg DirAcc | 58.3% | 41.7% | -16.7pp |
| # tickers LR better (MAPE) | — | 0/3 | — |

## Conclusion

❌ **Log-return normalization KHÔNG cải thiện**: MAPE tăng trung bình 3.74%. Kronos có thể đã handle price magnitude tốt ở tokenizer level. Khuyến nghị: thử fine-tune trực tiếp trên absolute prices.

## Full Comparison Table

| Ticker | ZS MAPE | LR MAPE | ΔMAPE | ZS Dir | LR Dir | ΔDir |
|--------|---------|---------|-------|--------|--------|------|
| VNM | 1.01% | 2.28% | ❌ +1.27% | 55% | 50% | ❌ -5pp |
| VRE | 5.27% | 7.04% | ❌ +1.77% | 50% | 40% | ❌ -10pp |
| VPB | 1.34% | 9.51% | ❌ +8.17% | 70% | 35% | ❌ -35pp |

## Top-5 Most Improved

| Ticker | ZS MAPE | LR MAPE | ΔMAPE | ΔDir |
|--------|---------|---------|-------|------|
| VNM | 1.01% | 2.28% | +1.27% | -5pp |
| VRE | 5.27% | 7.04% | +1.77% | -10pp |
| VPB | 1.34% | 9.51% | +8.17% | -35pp |

## Bottom-5 Most Degraded

| Ticker | ZS MAPE | LR MAPE | ΔMAPE | ΔDir |
|--------|---------|---------|-------|------|
| VPB | 1.34% | 9.51% | +8.17% | -35pp |
| VRE | 5.27% | 7.04% | +1.77% | -10pp |
| VNM | 1.01% | 2.28% | +1.27% | -5pp |

---
*Chart: `results/charts_logreturn/lr_comparison.png`*
*Predictions: `results/predictions_logreturn/`*