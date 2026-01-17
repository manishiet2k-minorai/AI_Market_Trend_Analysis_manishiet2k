import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Market Trend Predictor", layout="wide")

MODEL_PATH = "rf_trend_model.joblib"  # your saved model
DEFAULT_FEATURES_CSV = "spx_trend_features_2005_2025.csv"

@st.cache_resource
def load_model_bundle(path: str):
    bundle = joblib.load(path)
    return bundle["model"], bundle["features"]

def align_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[features].astype(float)

st.title("📈 AI Market Trend Analysis UI (5-day ahead)")

# Sidebar
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Bullish threshold", 0.05, 0.95, 0.50, 0.01)

# Load model
try:
    model, features = load_model_bundle(MODEL_PATH)
    st.sidebar.success(f"Loaded model: {MODEL_PATH}")
    st.sidebar.write("Features used:", features)
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    st.stop()

tab1, tab2 = st.tabs(["📁 Predict from CSV", "🧮 Manual input (single row)"])

# -------------------------
# TAB 1: CSV Upload Predict
# -------------------------
with tab1:
    st.subheader("Predict from CSV (recommended)")

    st.caption(
        "Upload a CSV that already contains the engineered feature columns "
        "(like spx_trend_features_2005_2025.csv)."
    )

    uploaded = st.file_uploader("Upload feature CSV", type=["csv"])

    if uploaded is None:
        st.info("No file uploaded. Using default feature dataset if available.")
        try:
            df = pd.read_csv(DEFAULT_FEATURES_CSV)
            st.success(f"Loaded default: {DEFAULT_FEATURES_CSV}")
        except Exception:
            st.warning("Default file not found. Upload a CSV to continue.")
            st.stop()
    else:
        df = pd.read_csv(uploaded)

    st.write("Preview:")
    st.dataframe(df.head(10), use_container_width=True)

    # If Date exists, parse it for nicer display
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    try:
        X = align_features(df, features)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Predict
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    out = df.copy()
    out["bullish_prob_5d"] = probs
    out["predicted_trend_5d"] = preds  # 1 = bullish, 0 = bearish

    st.subheader("Results")
    st.dataframe(out.tail(30), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Probability (last 200 rows)")
        st.line_chart(out["bullish_prob_5d"].tail(200))

    with col2:
        st.subheader("Predicted Trend (last 200 rows)")
        st.line_chart(out["predicted_trend_5d"].tail(200))

    # Download
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download predictions CSV",
        data=csv_bytes,
        file_name="predictions.csv",
        mime="text/csv",
    )

# ------------------------------
# TAB 2: Manual Single Row Input
# ------------------------------
with tab2:
    st.subheader("Manual input (single prediction)")

    st.caption("Enter feature values manually. Useful for quick testing.")

    # Build inputs dynamically for each feature
    input_vals = {}
    cols = st.columns(3)
    for i, f in enumerate(features):
        with cols[i % 3]:
            input_vals[f] = st.number_input(f, value=0.0, format="%.6f")

    if st.button("Predict"):
        row = pd.DataFrame([input_vals])
        p = model.predict_proba(row[features].astype(float))[:, 1][0]
        pred = int(p >= threshold)

        st.metric("Bullish Probability (5d ahead)", f"{p:.3f}")
        st.metric("Predicted Trend", "Bullish ✅" if pred == 1 else "Bearish ⚠️")
