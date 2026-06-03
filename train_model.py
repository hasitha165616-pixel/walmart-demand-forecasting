import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import shap
import os

print("=" * 60)
print("WALMART SALES FORECASTING — MODEL TRAINING")
print("=" * 60)

# ── 1. Load data ──────────────────────────────────────────────
# Place the 4 Kaggle CSVs in data/raw/ before running
# https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast

df_train    = pd.read_csv("data/raw/train.csv")
df_test     = pd.read_csv("data/raw/test.csv")
df_features = pd.read_csv("data/raw/features.csv")
df_stores   = pd.read_csv("data/raw/stores.csv")

df_train = df_train.merge(df_features, on=["Store", "Date", "IsHoliday"], how="left")
df_train = df_train.merge(df_stores,   on="Store", how="left")

print(f"✓ Training data loaded: {df_train.shape}")

# ── 2. Feature engineering ────────────────────────────────────
def feature_engineer(df):
    df = df.copy()
    df["Date"]     = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df["Year"]     = df["Date"].dt.year
    df["Month"]    = df["Date"].dt.month
    df["Week"]     = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"]  = df["Date"].dt.quarter
    df["DayOfYear"]= df["Date"].dt.dayofyear
    df["Week_sin"] = np.sin(2 * np.pi * df["Week"] / 52)
    df["Week_cos"] = np.cos(2 * np.pi * df["Week"] / 52)

    markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    df[markdown_cols] = df[markdown_cols].fillna(0)
    df["Total_MarkDown"] = df[markdown_cols].sum(axis=1)
    df["Has_Promotion"]  = (df["Total_MarkDown"] > 0).astype(int)

    for col in ["CPI", "Unemployment"]:
        df[col] = df.groupby("Store")[col].ffill().bfill()

    df["Type"]      = df["Type"].map({"A": 0, "B": 1, "C": 2})
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    return df

df_train = feature_engineer(df_train)
print("✓ Feature engineering complete")

# ── 3. Lag features ───────────────────────────────────────────
df_train = df_train.sort_values(["Store", "Dept", "Date"])

df_train["lag_1"] = df_train.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1)
df_train["lag_2"] = df_train.groupby(["Store", "Dept"])["Weekly_Sales"].shift(2)
df_train["lag_4"] = df_train.groupby(["Store", "Dept"])["Weekly_Sales"].shift(4)
df_train["lag_8"] = df_train.groupby(["Store", "Dept"])["Weekly_Sales"].shift(8)

df_train["rolling_mean_4"] = (
    df_train.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1).rolling(4).mean()
)
df_train["rolling_std_4"] = (
    df_train.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1).rolling(4).std()
)

df_train = df_train.dropna()
print(f"✓ Lag features created: {df_train.shape}")

# ── 4. Features & split ───────────────────────────────────────
FEATURES = [
    "Store", "Dept", "Size", "Type",
    "Temperature", "Fuel_Price",
    "CPI", "Unemployment",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "Total_MarkDown", "Has_Promotion",
    "IsHoliday",
    "Year", "Month", "Week", "Quarter", "DayOfYear",
    "Week_sin", "Week_cos",
    "lag_1", "lag_2", "lag_4", "lag_8",
    "rolling_mean_4", "rolling_std_4"
]
TARGET = "Weekly_Sales"

# Time-based split: last 20 weeks = validation
df_train = df_train.sort_values("Date")
cutoff   = df_train["Date"].quantile(0.8)
train    = df_train[df_train["Date"] <= cutoff]
val      = df_train[df_train["Date"] >  cutoff]

X_train, y_train = train[FEATURES], train[TARGET]
X_val,   y_val   = val[FEATURES],   val[TARGET]

print(f"✓ Train: {len(X_train):,}  |  Val: {len(X_val):,}")

# ── 5. Baseline ───────────────────────────────────────────────
baseline_mae = mean_absolute_error(y_val, np.full(len(y_val), y_train.mean()))
print(f"\nNaive baseline MAE: ${baseline_mae:,.2f}")

# ── 6. XGBoost (main model) ───────────────────────────────────
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
xgb_preds = xgb_model.predict(X_val)
xgb_mae   = mean_absolute_error(y_val, xgb_preds)
print(f"XGBoost MAE:        ${xgb_mae:,.2f}  ({((baseline_mae-xgb_mae)/baseline_mae*100):.1f}% improvement)")

# ── 7. Holiday model ──────────────────────────────────────────
train_hol = train[train["IsHoliday"] == 1]
val_hol   = val[val["IsHoliday"] == 1]

xgb_holiday = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
if len(train_hol) > 100:
    xgb_holiday.fit(train_hol[FEATURES], train_hol[TARGET])

# Hybrid predictions
hybrid_preds = xgb_preds.copy()
if len(val_hol) > 0:
    hol_idx = val[val["IsHoliday"] == 1].index
    hol_pos = [val.index.get_loc(i) for i in hol_idx]
    hybrid_preds[hol_pos] = xgb_holiday.predict(val_hol[FEATURES])

hybrid_mae = mean_absolute_error(y_val, hybrid_preds)
print(f"Hybrid MAE:         ${hybrid_mae:,.2f}  ({((baseline_mae-hybrid_mae)/baseline_mae*100):.1f}% improvement)")

# ── 8. Save val results for the app ──────────────────────────
val_results = val[["Store", "Dept", "Date", "IsHoliday", "Weekly_Sales"]].copy()
val_results["Predicted_Sales"] = hybrid_preds
val_results["Error"]           = val_results["Predicted_Sales"] - val_results["Weekly_Sales"]
val_results.to_csv("val_results.csv", index=False)
print("✓ Saved val_results.csv")

# ── 9. SHAP explainer ─────────────────────────────────────────
print("\nComputing SHAP values (this takes ~1 min)...")
explainer  = shap.TreeExplainer(xgb_model)
shap_sample = X_val.sample(min(2000, len(X_val)), random_state=42)
shap_values = explainer.shap_values(shap_sample)

# Global feature importance from SHAP
shap_importance = pd.DataFrame({
    "Feature":    FEATURES,
    "Importance": np.abs(shap_values).mean(axis=0)
}).sort_values("Importance", ascending=False)
shap_importance.to_csv("shap_importance.csv", index=False)
print("✓ Saved shap_importance.csv")

# ── 10. Save models ───────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(xgb_model,   "models/xgboost_model.pkl")
joblib.dump(xgb_holiday, "models/xgboost_holiday_model.pkl")
joblib.dump(explainer,   "models/explainer.pkl")
joblib.dump(FEATURES,    "models/features.pkl")

print("\n✅ Saved: models/xgboost_model.pkl")
print("✅ Saved: models/xgboost_holiday_model.pkl")
print("✅ Saved: models/explainer.pkl")
print("✅ Saved: models/features.pkl")
print("\n" + "=" * 60)
print(f"Final MAE improvement over baseline: {((baseline_mae-hybrid_mae)/baseline_mae*100):.1f}%")
print("=" * 60)