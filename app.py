import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from src.agent import RetailBankingAgent
from src.tools.eda_tool import run_eda, validate_dataset
from src.tools.explainability_tool import get_cross_sell_recommendations
from src.tools.kpi_tool import compute_business_kpis
from src.config import DEFAULT_SAMPLE_SIZE
from src.logger_setup import get_logger

logger = get_logger("app")

st.set_page_config(page_title="Retail Banking AI Agent: Segmentation & Personalization", layout="wide")

st.title("🏦 Retail Banking AI Agent: Customer Segmentation & Personalization")
st.markdown("Automated EDA, Behavioral Segmentation, Model Evaluation & Recommendations.")

# ---------------------------------------------------------------- Sidebar --
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.subheader("Performance")
use_full_dataset = st.sidebar.checkbox("Use full dataset (slower)", value=False)
sample_size = st.sidebar.number_input(
    "Sample size (rows) if not using full dataset",
    min_value=1000, max_value=1_000_000, value=DEFAULT_SAMPLE_SIZE, step=10000,
    help="The public Bank Customer Segmentation dataset has 1M+ rows. Sampling keeps EDA/clustering fast for a live demo."
)


@st.cache_data
def load_data(file):
    return pd.read_csv(file)


if uploaded_file is None:
    st.info("👈 Please upload a dataset (e.g. `bank_transactions.csv`) to begin.")
    st.stop()

with st.spinner("Loading dataset..."):
    try:
        df_raw = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")
        st.stop()

# ---------------------------------------------------------------- Validation
validation_issues = validate_dataset(df_raw)
if validation_issues:
    for issue in validation_issues:
        st.sidebar.warning(f"⚠️ {issue}")
    if df_raw.empty:
        st.error("Uploaded dataset is empty — nothing to analyze.")
        st.stop()

original_row_count = len(df_raw)
if not use_full_dataset and original_row_count > sample_size:
    df_raw = df_raw.sample(n=int(sample_size), random_state=42).reset_index(drop=True)
    st.sidebar.success(f"Loaded {original_row_count:,} records — sampled down to {len(df_raw):,} for speed.")
else:
    st.sidebar.success(f"Loaded {len(df_raw):,} records successfully!")

if api_key:
    os.environ["GEMINI_API_KEY"] = api_key

# ---------------------------------------------------------------- Agent init
if "agent" not in st.session_state or st.sidebar.button("Re-initialize Agent"):
    with st.status("Initializing Retail Banking Agent...", expanded=True) as status:
        st.write("Loading dataset into agent...")
        try:
            agent = RetailBankingAgent(df_raw)
        except Exception as e:
            status.update(label="Agent initialization failed", state="error")
            st.error(f"Could not initialize the agent — check your Gemini API key. Details: {e}")
            st.stop()

        st.write("Running default preprocessing + segmentation pipeline (KMeans)...")
        try:
            agent.run_default_pipeline(method="kmeans")
        except Exception as e:
            status.update(label="Default pipeline failed", state="error")
            st.error(f"Default segmentation pipeline failed: {e}")
            st.stop()

        for step in agent.execution_log:
            st.write(f"✅ {step}")
        status.update(label="Agent ready", state="complete")

    st.session_state.agent = agent

agent = st.session_state.agent

# ---------------------------------------------------------------- KPI Dashboard
st.markdown("### 📌 Business KPI Dashboard")
kpis = compute_business_kpis(agent.segmented_df) if agent.segmented_df is not None else {}
if kpis:
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Customers", f"{kpis.get('total_customers', 0):,}")
    k2.metric("🥇 Priority", f"{kpis.get('priority_customers', 0):,}")
    k3.metric("🥈 Regular", f"{kpis.get('regular_customers', 0):,}")
    k4.metric("💤 Dormant", f"{kpis.get('dormant_customers', 0):,}")
    k5.metric("Avg Balance", f"₹{kpis.get('avg_balance', 0):,.0f}")
    k6.metric("Cross-Sell Potential", f"{kpis.get('cross_sell_potential', 0):,}")
else:
    st.info("KPIs will appear once segmentation has run.")

st.markdown("---")

# ---------------------------------------------------------------- Tabs
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
        "\"Show only the priority ones\" (after segmenting), "
        "\"Compare clustering methods\", \"What are the business KPIs?\""
    )
    user_query = st.text_input("Type your analytical query:", placeholder="e.g., On what basis were priority customers selected?")

    if st.button("Send Query") and user_query:
        with st.spinner("Agent processing..."):
            result = agent.process_query(user_query)

        st.markdown("### Agent Response")
        st.write(result["answer"])

        with st.expander("🔍 Agent reasoning / execution log"):
            for step in result["log"]:
                st.text(f"• {step}")

        st.download_button(
            "📥 Download this response as a report (.txt)",
            data=f"Query: {user_query}\n\nAnswer:\n{result['answer']}\n\nExecution log:\n" + "\n".join(result["log"]),
            file_name="agent_response_report.txt",
            mime="text/plain",
        )

# Tab 2: Dynamic EDA
with tab2:
    st.subheader("Automated Exploratory Data Analysis")
    eda_summary = run_eda(agent.raw_df)

    col1, col2 = st.columns(2)
    col1.metric("Total Records Analyzed", eda_summary["total_rows"])
    col2.metric("Total Features", eda_summary["total_columns"])

    st.write("#### Missing Values Summary", eda_summary["missing_values"])
    st.dataframe(pd.DataFrame(eda_summary["numeric_summary"]).T)

# Tab 3: Customer Segments & Export
with tab3:
    st.subheader("Customer Segments & Data Export")
    seg_df = agent.segmented_df

    if seg_df is not None:
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                label="📥 Download Segmented Customers (CSV)",
                data=seg_df.to_csv(index=False).encode('utf-8'),
                file_name="segmented_customers.csv",
                mime="text/csv",
                type="primary"
            )
        with dl2:
            personas_summary = json.dumps(compute_business_kpis(seg_df), indent=2)
            st.download_button(
                label="📥 Download Persona/KPI Summary (JSON)",
                data=personas_summary,
                file_name="segment_kpi_summary.json",
                mime="application/json",
            )

        st.dataframe(seg_df.head(100))

        st.write("#### Segment Distribution")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.countplot(data=seg_df, x="Segment", hue="Segment", palette="Blues_r", legend=False, ax=ax)
        st.pyplot(fig)
    else:
        st.info("No segmentation has been run yet.")

# Tab 4: Model Evaluation
with tab4:
    st.subheader("Clustering Model Performance Evaluation")
    eval_data = agent.evaluation_metrics

    if eval_data:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Segmentation Method", str(eval_data.get("method", "N/A")).upper())
        col2.metric("Silhouette Score", eval_data.get("silhouette_score", "N/A"))
        col3.metric("Davies-Bouldin", eval_data.get("davies_bouldin_score", "N/A"))
        col4.metric("Calinski-Harabasz", eval_data.get("calinski_harabasz_score", "N/A"))

        st.info(f"**Model Quality Assessment:** {eval_data.get('quality', 'Evaluated based on customer feature distributions.')}")

        if "segment_distribution" in eval_data:
            st.write("#### Segment Distribution")
            st.json(eval_data["segment_distribution"])

        if st.button("🔬 Compare KMeans vs Hierarchical vs DBSCAN"):
            with st.spinner("Running all three clustering methods..."):
                from src.tools.segmentation_tool import compare_clustering_methods
                comparison = compare_clustering_methods(agent.processed_df)
            st.json(comparison)
    else:
        st.write("Run segmentation to display evaluation metrics.")

# Tab 5: Recommendations
with tab5:
    st.subheader("💡 Cross-Selling & Personalization Engine")
    st.write("Select a customer segment tier below to inspect AI-generated recommendations and marketing strategies.")

    selected_segment = st.selectbox("Select Customer Segment Tier:", ["Priority", "Regular", "Dormant"])

    if st.button("Generate Strategy"):
        strategy_output = get_cross_sell_recommendations(selected_segment)
        st.success(f"Strategy Generated for **{selected_segment}** Tier:")
        st.markdown(strategy_output)

    st.markdown("---")
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