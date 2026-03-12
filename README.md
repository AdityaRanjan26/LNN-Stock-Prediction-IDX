# LNN Stock Prediction — IDX Market
### TENCON 2026 | IEEE Region 10 Conference

> **Liquid Neural Networks for Stock Price Direction Prediction on the Indonesian Stock Exchange (IDX)**

---

## Results

| Model | Test DA | Std |
|---|---|---|
| **LNN Ensemble** | **58.77%** | ±4.87% |
| Transformer | 56.40% | ±4.71% |
| GRU | 55.98% | ±4.57% |
| LSTM | 54.57% | ±7.01% |
| Random Baseline | 50.00% | — |

LNN outperforms all baselines by **+2.4% to +4.2%** on directional accuracy.

---

## Architecture

```
IDX Stock Data (yfinance, 7yr, 45 stocks)
        ↓
Technical Feature Engineering (47 features)
        ↓
Mutual Information Feature Selection
        ↓
RobustScaler (fit on train only)
        ↓
CNN Feature Extractor
        ↓
Stacked Wired LTC (NCP topology)
        ↓
Multi-Head Attention
        ↓
Direction Classifier (UP / DOWN)
```

**LNN Cell:** Liquid Time-Constant (LTC) with NCP-inspired wiring
- Inter neurons: 96
- Command neurons: 48
- Motor neurons: 24
- ODE unfolds: 3
- Ensemble: 3 seeds (42, 123, 777)

---

## Dataset

- **Market:** Indonesian Stock Exchange (IDX)
- **Universe:** 45 stocks across 8 sectors (Banking, Telecom, Consumer, Industrial, Energy, Property, Healthcare, Finance)
- **Period:** 7 years historical data
- **Macro indicators:** IHSG, USD/IDR, Gold, VIX, DXY, Coal, Nickel
- **Split:** 70% train / 15% val / 15% test (chronological)
- **Samples:** ~47,000 training sequences

---

## Features

47 features selected via Mutual Information from:
- Price returns (1d, 2d, 3d, 5d, 10d, 20d)
- Moving averages (MA5, MA10, MA20, MA50)
- EMA (5, 12, 26)
- MACD + Signal + Histogram
- RSI (7, 14, 21)
- Bollinger Bands (10, 20)
- ATR, Volatility
- Volume features (OBV, ratio)
- Stochastic oscillator
- Sector momentum
- Macro indicators
- Calendar features (sin/cos encoding)

---

## Project Structure

```
LNN-Stock-Prediction-IDX/
├── train.py              ← main entry point
├── config.py             ← all hyperparameters
├── models/
│   ├── lnn.py            ← LNN architecture (LTC + NCP)
│   └── baselines.py      ← LSTM, GRU, Transformer
├── utils/
│   ├── data_loader.py    ← yfinance + feature engineering
│   ├── features.py       ← MI selection + scaling
│   ├── dataset.py        ← PyTorch Dataset
│   ├── trainer.py        ← training loop
│   ├── metrics.py        ← DA, F1, AUC, stat tests
│   └── plotter.py        ← visualization
├── saved_models/         ← trained model weights
├── results/              ← JSON results
└── plots/                ← training curves, comparisons
```

---

## Setup

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/LNN-Stock-Prediction-IDX.git
cd LNN-Stock-Prediction-IDX

# Install dependencies
pip install torch yfinance pandas numpy scikit-learn matplotlib scipy

# Train
python train.py --mode train --epochs 150

# Evaluate saved models
python train.py --mode evaluate

# Generate comparison plots
python train.py --mode compare
```

---

## Requirements

```
torch>=2.0
yfinance>=0.2
pandas>=1.5
numpy>=1.24
scikit-learn>=1.2
matplotlib>=3.6
scipy>=1.10
```

---

## Key Findings

- LNN with NCP wiring outperforms standard RNNs on financial time series
- Liquid time-constants allow adaptive processing of irregular market dynamics
- Ensemble of 3 seeds reduces variance significantly (±4.87% vs ±7.01% for LSTM)
- Confidence-filtered predictions achieve higher DA at the cost of coverage
- Indonesian market shows strong sector momentum effects

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{lnn_idx_tencon2026,
  title     = {Liquid Neural Networks for Stock Price Direction Prediction 
               on the Indonesian Stock Exchange},
  booktitle = {TENCON 2026 - IEEE Region 10 Conference},
  year      = {2026}
}
```

---

## License

MIT License — free to use for research and educational purposes.

---

*Submitted to TENCON 2026 — IEEE Region 10 Conference, Bali, Indonesia*
