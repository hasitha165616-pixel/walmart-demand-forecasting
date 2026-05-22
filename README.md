# Walmart Sales Forecasting
**Predicting weekly retail demand with XGBoost + Explainable AI**

---

## 🎯 Problem

Walmart needs accurate sales forecasts to:
- Avoid stockouts (lost revenue)
- Prevent overstock (wasted capital)
- Plan inventory across 45 stores, 81 departments
- Meet investor expectations (stock price impact)

**Bad forecasts = millions in losses.**

---
## 📊 Solution

Built a machine learning forecasting system that predicts weekly sales **16.5% better** than baseline.

| Model | MAE | Improvement |
|---|---|---|
| Naive Baseline | $1,721 | - |
| Random Forest | $1,469 | 14.6% |
| **XGBoost** | **$1,437** | **16.5%** ✅ |

---

## 🔍 Key Findings

**1. Lag features dominate (68% importance)**
- Last week's sales predict this week better than any external factor
- Weather, economy, promotions = minimal impact

**2. Department 92 is a problem child**
- Appears in 5 of top 10 hardest-to-forecast combinations
- Average error: $10,000+/week
- **Recommendation:** Increase safety stock 25-30%

**3. Holidays are 21% harder to forecast**
- Holiday MAE: $1,751 vs Non-Holiday: $1,422
- Extreme spikes (Thanksgiving +60%, Black Friday +80%)
- **Recommendation:** Accept higher buffer stock during holidays

**4. High-error stores identified**
- Stores 20, 14, 13 account for 35% of total error
- **Recommendation:** Build store-specific models

---

## 💼 Business Impact

**Potential value:** ~$7M annual savings across all stores  
*(Based on reduced stockouts + overstock costs)*

**Immediate actions:**
1. Deploy XGBoost for weekly forecasting
2. Increase safety stock for Dept 92 combinations
3. Add daily inventory reviews for top-error stores

---

## 🛠️ How It Works

### Data
- 421,570 weekly observations (2010-2012)
- 45 stores, 81 departments
- Features: Sales history, promotions, store attributes, economic indicators

### Feature Engineering
- **Lag features:** `lag_1`, `lag_2`, `lag_4`, `lag_8`
- **Rolling stats:** 4-week moving average, std dev
- **Time encoding:** Cyclical week/month encoding
- **Promotions:** Total markdown, promo flag

### Models
- XGBoost (300 trees, learning rate 0.05)
- SHAP for explainability

---

## 📈 Visualizations

### SHAP Feature Importance
![SHAP](images/shap_summary.png)

### Model Comparison
![Comparison](images/model_comparison.png)

### Error Analysis
![Error](images/error_analysis.png)

---

## 🚀 Usage
```bash
# Install dependencies
pip install -r requirements.txt
# Download data from Kaggle
# https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast
# Place CSVs in data/raw/

# Run analysis
cd notebooks
python walmart_forecast_upgraded.py
```

**Output:**
- Performance metrics
- SHAP plots
- Business insights
- Saved models

---

## 📁 Project Structure
```
walmart-sales-forecast/
├── data/raw/              # Kaggle data
├── images/                # Visualizations
├── notebooks/             # Analysis script
├── models/                # Saved models
├── requirements.txt
└── README.md
```

---

## 🔮 Future Improvements

- [ ] Deploy as Streamlit dashboard
- [ ] Add prediction intervals (uncertainty quantification)
- [ ] Department-specific models for Dept 92
- [ ] External data: weather, local events

---

## 🎓 Tech Stack

Python • XGBoost • SHAP • pandas • scikit-learn • matplotlib

---

## 🙏 Data Source

[Walmart Recruiting - Store Sales Forecasting (Kaggle)](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast)

---

**⭐ Star this repo if you found it helpful!**