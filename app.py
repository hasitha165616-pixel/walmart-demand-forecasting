import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Walmart Sales Forecasting",
    page_icon="🛒",
    layout="wide",
)

# ── Load artifacts ────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model      = joblib.load("models/xgboost_model.pkl")
    explainer  = joblib.load("models/explainer.pkl")
    features   = joblib.load("models/features.pkl")
    return model, explainer, features

@st.cache_data
def load_data():
    val      = pd.read_csv("val_results.csv", parse_dates=["Date"])
    shap_imp = pd.read_csv("shap_importance.csv")
    return val, shap_imp

model, explainer, FEATURES = load_artifacts()
val_results, shap_importance = load_data()

stores = sorted(val_results["Store"].unique())
depts  = sorted(val_results["Dept"].unique())

# ── Header ────────────────────────────────────────────────────
st.title("🛒 Walmart Sales Forecasting Dashboard")
st.caption("XGBoost + Hybrid Holiday Model · 16.1% improvement over baseline · Trained on 421K weekly records")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Store & Department Forecast", "📊 Model Performance", "🧠 Feature Importance"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — Store / Dept Forecast
# ══════════════════════════════════════════════════════════════
with tab1:
    col_s, col_d = st.columns(2)
    with col_s:
        selected_store = st.selectbox("Select Store", stores)
    with col_d:
        available_depts = sorted(
            val_results[val_results["Store"] == selected_store]["Dept"].unique()
        )
        selected_dept = st.selectbox("Select Department", available_depts)

    subset = val_results[
        (val_results["Store"] == selected_store) &
        (val_results["Dept"]  == selected_dept)
    ].sort_values("Date")

    if len(subset) < 5:
        st.warning("Not enough data for this store/department combination. Try another.")
    else:
        # ── Metrics ───────────────────────────────────────────
        mae       = np.abs(subset["Error"]).mean()
        accuracy  = 100 - (np.abs(subset["Error"]) / subset["Weekly_Sales"] * 100).mean()
        avg_sales = subset["Weekly_Sales"].mean()
        holiday_mae     = np.abs(subset[subset["IsHoliday"] == 1]["Error"]).mean() if subset["IsHoliday"].sum() > 0 else 0
        non_holiday_mae = np.abs(subset[subset["IsHoliday"] == 0]["Error"]).mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Weekly Sales",    f"${avg_sales:,.0f}")
        m2.metric("Forecast MAE",        f"${mae:,.0f}")
        m3.metric("Forecast Accuracy",   f"{accuracy:.1f}%")
        m4.metric("Holiday MAE",         f"${holiday_mae:,.0f}" if holiday_mae else "N/A")

        st.divider()

        # ── Actual vs Predicted chart ─────────────────────────
        st.subheader(f"Store {selected_store} · Dept {selected_dept} — Actual vs Predicted")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(subset["Date"], subset["Weekly_Sales"],
                label="Actual Sales", linewidth=2, color="#1f77b4")
        ax.plot(subset["Date"], subset["Predicted_Sales"],
                label="Predicted Sales", linewidth=2, linestyle="--", color="#ff7f0e")
        ax.fill_between(subset["Date"],
                        subset["Weekly_Sales"], subset["Predicted_Sales"],
                        alpha=0.15, color="gray", label="Forecast Error")

        # Mark holidays
        holiday_dates = subset[subset["IsHoliday"] == 1]["Date"]
        for hd in holiday_dates:
            ax.axvline(hd, color="red", alpha=0.25, linewidth=1)
        if len(holiday_dates) > 0:
            ax.axvline(holiday_dates.iloc[0], color="red", alpha=0.25,
                       linewidth=1, label="Holiday week")

        ax.set_xlabel("Date")
        ax.set_ylabel("Weekly Sales ($)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.caption("Red vertical lines = holiday weeks")

        # ── Error distribution ────────────────────────────────
        st.subheader("Forecast Error Distribution")
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.hist(subset["Error"], bins=30, color="#4ecdc4", edgecolor="black", alpha=0.8)
        ax2.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero error")
        ax2.set_xlabel("Prediction Error ($)")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Overall Model Performance")

    # Overall metrics
    overall_mae      = np.abs(val_results["Error"]).mean()
    overall_accuracy = 100 - (np.abs(val_results["Error"]) / val_results["Weekly_Sales"] * 100).mean()
    hol_mask         = val_results["IsHoliday"] == 1
    hol_mae          = np.abs(val_results[hol_mask]["Error"]).mean()
    non_hol_mae      = np.abs(val_results[~hol_mask]["Error"]).mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall MAE",         f"${overall_mae:,.0f}")
    c2.metric("Overall Accuracy",    f"{overall_accuracy:.1f}%")
    c3.metric("Holiday MAE",         f"${hol_mae:,.0f}")
    c4.metric("Non-Holiday MAE",     f"${non_hol_mae:,.0f}")

    st.divider()

    col_l, col_r = st.columns(2)

    # Model comparison bar chart
    with col_l:
        st.subheader("Model Comparison")
        baseline_mae = 1720.75
        rf_mae       = 1468.70
        xgb_mae_val  = 1437.40
        hybrid_mae   = 1443.42

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        models = ["Naive\nBaseline", "Random\nForest", "XGBoost", "Hybrid\n(Holiday)"]
        maes   = [baseline_mae, rf_mae, xgb_mae_val, hybrid_mae]
        colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4"]
        bars   = ax3.bar(models, maes, color=colors, edgecolor="black", linewidth=1)
        for bar in bars:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h,
                     f"${h:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax3.set_ylabel("Mean Absolute Error ($)")
        ax3.set_title("MAE by Model")
        ax3.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    # Top hardest store-dept combos
    with col_r:
        st.subheader("Top 10 Hardest to Forecast")
        hard = (
            val_results.groupby(["Store", "Dept"])
            .apply(lambda x: np.abs(x["Error"]).mean())
            .reset_index(name="MAE")
            .sort_values("MAE", ascending=False)
            .head(10)
        )
        hard["MAE"] = hard["MAE"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(hard.reset_index(drop=True), use_container_width=True, hide_index=True)
        st.caption("These store-dept combos need higher safety stock buffers.")

    st.divider()

    # Holiday vs non-holiday insight
    st.subheader("Holiday vs Non-Holiday Forecast Difficulty")
    fig4, ax4 = plt.subplots(figsize=(6, 3))
    ax4.bar(["Non-Holiday", "Holiday"], [non_hol_mae, hol_mae],
            color=["#45b7d1", "#ff6b6b"], edgecolor="black", linewidth=1)
    for i, v in enumerate([non_hol_mae, hol_mae]):
        ax4.text(i, v, f"${v:,.0f}", ha="center", va="bottom", fontweight="bold")
    ax4.set_ylabel("Mean Absolute Error ($)")
    ax4.set_title("Holiday weeks are harder to forecast")
    ax4.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 3 — Feature Importance
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("SHAP Feature Importance (Global)")
    st.caption("Average absolute SHAP value across 2,000 validation samples — shows which features matter most to the model.")

    top_n = st.slider("Show top N features", 5, 29, 15)
    top_features = shap_importance.head(top_n)

    fig5, ax5 = plt.subplots(figsize=(9, top_n * 0.45 + 1))
    bars = ax5.barh(top_features["Feature"][::-1],
                    top_features["Importance"][::-1],
                    color="#45b7d1", edgecolor="black", linewidth=0.5)
    ax5.set_xlabel("Mean |SHAP Value|")
    ax5.set_title(f"Top {top_n} Features by SHAP Importance")
    ax5.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

    st.divider()
    st.subheader("Key Findings")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Lag features dominate**
- `lag_1`, `lag_2`, `lag_4`, `lag_8` collectively account for ~76% of model importance
- Last week's sales is the single strongest predictor

**Holidays are 21% harder to forecast**
- Holiday MAE is significantly higher than non-holiday
- Hybrid model addresses this with a dedicated holiday XGBoost
        """)
    with col2:
        st.markdown("""
**Business recommendations**
- 📦 Increase safety stock 25–30% for top-10 high-error store-dept combos
- 📅 Run daily inventory checks during holiday weeks
- 🏪 Consider store-specific models for highest-error stores
- 📊 Lag features > external factors (weather, CPI, unemployment)
        """)