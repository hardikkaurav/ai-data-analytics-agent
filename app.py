import streamlit as st
import pandas as pd

from src.data_loader import load_data
from src.eda import basic_eda
from src.trend_analysis import sales_trend_analysis
from src.anomaly_detection import anomaly_summary
from src.insight_generator import build_insight_prompt
from src.llm_client import generate_insights

# --------------------------------------------------
# App Config
# --------------------------------------------------
st.set_page_config(
    page_title="AI Data Analytics Agent",
    layout="wide"
)

st.title("📊 AI Data Analytics Agent")
st.caption("Upload a dataset and get automated insights using ML + GenAI")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

use_sample = st.sidebar.checkbox("Use sample sales dataset")

# --------------------------------------------------
# Load Data
# --------------------------------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
elif use_sample:
    df = load_data("data/sales_data.csv")
else:
    st.info("Upload a CSV file or select sample dataset.")
    st.stop()

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# Run Analytics
# --------------------------------------------------
with st.spinner("Running automated analysis..."):
    eda = basic_eda(df)
    trends = sales_trend_analysis(df)
    anomalies = anomaly_summary(df)

# --------------------------------------------------
# Display EDA
# --------------------------------------------------
st.subheader("🔍 Exploratory Data Analysis")

col1, col2 = st.columns(2)
col1.metric("Rows", eda["rows"])
col2.metric("Columns", eda["columns"])

if eda["missing_values"]:
    st.warning("Missing Values Detected")
    st.json(eda["missing_values"])
else:
    st.success("No missing values detected")

# --------------------------------------------------
# Trend Analysis
# --------------------------------------------------
st.subheader("📈 Trend Analysis")

st.write(f"**Overall Trend:** {trends.get('trend')}")
st.write(f"**Average Growth Rate:** {trends.get('average_growth_rate')}")

st.dataframe(pd.DataFrame(trends["time_series"]))

# --------------------------------------------------
# Anomaly Detection
# --------------------------------------------------
st.subheader("🚨 Anomaly Detection")

st.write("**Z-score anomalies:**", anomalies["zscore_anomalies_count"])
st.write("**Isolation Forest anomalies:**", anomalies["isolation_forest_anomalies_count"])

if anomalies.get("top_zscore_anomalies"):
    st.write("Top Z-score anomalies")
    st.json(anomalies["top_zscore_anomalies"])

# --------------------------------------------------
# GenAI Insights
# --------------------------------------------------
st.subheader("🧠 AI-Generated Insights")

if st.button("Generate Insights"):
    with st.spinner("Generating insights using GenAI..."):
        prompt = build_insight_prompt(eda, trends, anomalies)
        insights = generate_insights(prompt)

    st.markdown(insights)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "⚠️ Insights are generated based on data patterns and should be used for decision support only."
)
