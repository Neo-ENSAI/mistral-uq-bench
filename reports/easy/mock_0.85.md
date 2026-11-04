## ⚡ Mistral-UQ-Bench — Automated Statistical CI Report

**Status:** ⚠️ SHARPNESS ALERT
**Reason:** Conformal sets too large (2.93 > 2.2).

| Metric | Value | Target / Threshold |
| :--- | :--- | :--- |
| **Model Backend** | `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` | `mock` |
| **Conformal Method** | `LAC` ($\hat{q} = 0.9425$) | Split Conformal |
| **Empirical Coverage** | **94.00%** | $\ge 95.0\%$ |
| **Mean Prediction Set Size** | **2.93** | Reference: ≤ 2.20 (needs a ~70%-accuracy well-calibrated model) |
| **Model Accuracy (D_test)** | **83.3%** | Context for set size |
| **Set Size — model correct** | 2.92 | Sharpness when confident |
| **Set Size — model wrong** | 2.96 | Honesty when lost |
| **Expected Calibration Error (ECE)** | `0.1058` | Lower is better |
| **Martingale Drift Alert** | `False` | `False` ($M_t < 20$) |
| **Mean Throughput** | `116.7 tok/s` | - |
