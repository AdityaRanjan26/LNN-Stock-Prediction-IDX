# LNN Stock Prediction — IDX Market
### Liquid Neural Networks for Stock Price Direction Prediction

> Applying **Liquid Neural Networks (LNN)** with Neural Circuit Policy (NCP) wiring to predict stock price direction on the Indonesian Stock Exchange (IDX).

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
Wired LTC (NCP topology)
        ↓
Multi-Head Attention
        ↓
Direction Classifier (UP / DOWN)
```

**LNN Cell:** Liquid Time-Constant (LTC) with NCP-inspired wiring
- Inter neurons: 96 | Command neurons: 48 | Motor neurons: 24
- ODE unfolds: 3
- Ensemble: 3 seeds (42, 123, 777)

---

## Dataset

- **Market:** Indonesian Stock Exchange (IDX)
- **Universe:** 45 stocks across 8 sectors
- **Period:** 7 years historical data
- **Macro:** IHSG, USD/IDR, Gold, VIX, DXY, Coal, Nickel
- **Split:** 70% train / 15% val / 15% test (chronological)
- **Samples:** ~47,000 training sequences

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
├── saved_models/         ← scaler + feature columns
├── results/              ← JSON results
└── plots/                ← training curves
```

---

## Setup

```bash
git clone https://github.com/AdityaRanjan26/LNN-Stock-Prediction-IDX.git
cd LNN-Stock-Prediction-IDX
pip install torch yfinance pandas numpy scikit-learn matplotlib scipy

# Train from scratch
python train.py --mode train --epochs 150

# Evaluate
python train.py --mode evaluate
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

- LNN with NCP wiring outperforms LSTM/GRU/Transformer on financial time series
- Liquid time-constants allow adaptive processing of irregular market dynamics
- Ensemble of 3 seeds reduces variance (±4.87% vs ±7.01% for LSTM)
- Confidence-filtered predictions achieve higher DA at reduced coverage

---

## License

MIT License — free to use for research and educational purposes.
