import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.agent import RetailBankingAgent
from src.tools.eda_tool import run_eda
from src.tools.explainability_tool import get_cross_sell_recommendations
from src.config import DEFAULT_SAMPLE_SIZE

st.set_page_config(page_title="Retail Banking AI Agent: Segmentation & Personalization", layout="wide")

st.title("🏦 Retail Banking AI Agent: Customer Segmentation & Personalization")
st.markdown("Automated EDA, Behavioral Segmentation, Model Evaluation & Recommendations.")

# Sidebar Configuration
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.subheader("Performance")
use_full_dataset = st.sidebar.checkbox("Use full dataset (slower)", value=False)
sample_size = st.sidebar.number_input(
    "Sample size (rows) if not using full dataset",
    min_value=1000,
    max_value=1_000_000,
    value=DEFAULT_SAMPLE_SIZE,
    step=10000,
    help="The public Bank Customer Segmentation dataset has 1M+ rows. "
         "Sampling keeps EDA/clustering fast for a live demo."
)


@st.cache_data
def load_data(file):
    return pd.read_csv(file)


if uploaded_file is not None:
    with st.spinner("Loading dataset..."):
        df_raw = load_data(uploaded_file)

    original_row_count = len(df_raw)
    if not use_full_dataset and original_row_count > sample_size:
        df_raw = df_raw.sample(n=sample_size, random_state=42).reset_index(drop=True)
        st.sidebar.success(f"Loaded {original_row_count:,} records — sampled down to {len(df_raw):,} for speed.")
    else:
        st.sidebar.success(f"Loaded {len(df_raw):,} records successfully!")
else:
    st.info("👈 Please upload a dataset (e.g. `bank_transactions.csv`) to begin.")
    st.stop()

if api_key:
    os.environ["GEMINI_API_KEY"] = api_key

# Initialize Agent
if "agent" not in st.session_state or st.sidebar.button("Re-initialize Agent"):
    agent = RetailBankingAgent(df_raw)
    with st.spinner("Processing data & running default segmentation..."):
        agent.run_default_pipeline(method="kmeans")  # Runs KMeans to calculate evaluation metrics
    st.session_state.agent = agent

agent = st.session_state.agent

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Agent Chat",
    "📊 Dynamic EDA",
    "🎯 Customer Segments",
    "📈 Model Evaluation",
    "💡 Recommendations"
])

# Tab 1: Agent Chat UI
with tab1:
    st.subheader("Ask the Banking Analytics Agent")
    st.caption(
        "Try: \"On what basis were priority customers selected?\", "
        "\"Segment customers using kmeans with 4 clusters\", "
        "\"Which regular customers can be converted to priority customers?\", "
        "\"Is customer ID C5841053 suspicious... I mean, what segment are they in and why?\""
    )
    user_query = st.text_input("Type your analytical query:", placeholder="e.g., On what basis were priority customers selected?")

    if st.button("Send Query") and user_query:
        with st.spinner("Agent processing..."):
            response = agent.process_query(user_query)
            st.markdown("### Agent Response")
            st.write(response)

# Tab 2: Dynamic EDA
with tab2:
    st.subheader("Automated Exploratory Data Analysis")
    eda_summary = run_eda(agent.raw_df)

    col1, col2 = st.columns(2)
    col1.metric("Total Records Analyzed", eda_summary["total_rows"])
    col2.metric("Total Features", eda_summary["total_columns"])

    st.write("#### Missing Values Summary", eda_summary["missing_values"])
    st.dataframe(pd.DataFrame(eda_summary["numeric_summary"]).T)

# Tab 3: Customer Segments & Export CSV
with tab3:
    st.subheader("Customer Segments & Data Export")
    seg_df = agent.segmented_df

    if seg_df is not None:
        st.download_button(
            label="📥 Download Segmented Customers CSV",
            data=seg_df.to_csv(index=False).encode('utf-8'),
            file_name="segmented_customers.csv",
            mime="text/csv",
            type="primary"
        )

        st.dataframe(seg_df.head(100))

        st.write("#### Segment Distribution")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.countplot(data=seg_df, x="Segment", hue="Segment", palette="Blues_r", legend=False, ax=ax)
        st.pyplot(fig)

# Tab 4: Model Evaluation
with tab4:
    st.subheader("Clustering Model Performance Evaluation")
    eval_data = agent.evaluation_metrics

    if eval_data:
        col1, col2, col3 = st.columns(3)
        col1.metric("Segmentation Method", str(eval_data.get("method", "N/A")).upper())
        col2.metric("Silhouette Score", eval_data.get("silhouette_score", "N/A"))
        col3.metric("Clustering Inertia", eval_data.get("inertia", "N/A"))

        st.info(f"**Model Quality Assessment:** {eval_data.get('quality', 'Evaluated based on customer feature distributions.')}")

        if "segment_distribution" in eval_data:
            st.write("#### Segment Distribution")
            st.json(eval_data["segment_distribution"])
    else:
        st.write("Run K-Means segmentation to display evaluation metrics.")

# Tab 5: Dynamic Recommendations UI
with tab5:
    st.subheader("💡 Cross-Selling & Personalization Engine")
    st.write("Select a customer segment tier below to inspect AI-generated recommendations and marketing strategies.")

    selected_segment = st.selectbox(
        "Select Customer Segment Tier:",
        ["Priority", "Regular", "Dormant"]
    )

    if st.button("Generate Strategy"):
        strategy_output = get_cross_sell_recommendations(selected_segment)
        st.success(f"Strategy Generated for **{selected_segment}** Tier:")
        st.markdown(strategy_output)

    st.markdown("---")

    # Overview Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("### 🥇 Priority")
        st.caption("Target: High Wealth")
        st.markdown("- Wealth Management\n- Premium Travel Cards")
    with col2:
        st.write("### 🥈 Regular")
        st.caption("Target: Daily Banking")
        st.markdown("- Personal Loans\n- Auto Loans")
    with col3:
        st.write("### 💤 Dormant")
        st.caption("Target: Inactive Accounts")
        st.markdown("- UPI Promos\n- Zero Fee Accounts")