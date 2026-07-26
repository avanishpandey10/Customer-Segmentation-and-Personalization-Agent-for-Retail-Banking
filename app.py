import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from src.agent import RetailBankingAgent
from src.tools.eda_tool import run_eda, validate_dataset, generate_eda_summary
from src.tools.explainability_tool import (
    get_cross_sell_recommendations,
    generate_customer_personas,
    generate_segment_insights,
    get_retention_strategies,
)
from src.tools.kpi_tool import compute_business_kpis
from src.tools.feature_engineering import select_most_important_features
from src.config import DEFAULT_SAMPLE_SIZE
from src.logger_setup import get_logger

logger = get_logger("app")

st.set_page_config(
    page_title="Retail Banking AI Agent: Segmentation & Personalization",
    layout="wide",
)

st.title("🏦 Retail Banking AI Agent: Customer Segmentation & Personalization")
st.markdown(
    "Automated EDA, Behavioral Segmentation, Model Evaluation, "
    "Persona Generation & Retention Strategies."
)

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.subheader("Performance")
use_full_dataset = st.sidebar.checkbox("Use full dataset (slower)", value=False)
sample_size = st.sidebar.number_input(
    "Sample size (rows)",
    min_value=1000, max_value=1_000_000, value=DEFAULT_SAMPLE_SIZE, step=10000,
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
        st.error(f"Could not read file: {e}")
        st.stop()

validation_issues = validate_dataset(df_raw)
if validation_issues:
    for issue in validation_issues:
        st.sidebar.warning(f"⚠️ {issue}")
    if df_raw.empty:
        st.error("Uploaded dataset is empty.")
        st.stop()

original_row_count = len(df_raw)
if not use_full_dataset and original_row_count > sample_size:
    df_raw = df_raw.sample(n=int(sample_size), random_state=42).reset_index(drop=True)
    st.sidebar.success(f"Loaded {original_row_count:,} records — sampled to {len(df_raw):,}.")
else:
    st.sidebar.success(f"Loaded {len(df_raw):,} records.")

if api_key:
    os.environ["GEMINI_API_KEY"] = api_key

# ── Agent Init ────────────────────────────────────────────────────────
if "agent" not in st.session_state or st.sidebar.button("Re-initialize Agent"):
    with st.status("Initializing Retail Banking Agent...", expanded=True) as status:
        try:
            agent = RetailBankingAgent(df_raw)
        except Exception as e:
            status.update(label="Agent initialization failed", state="error")
            st.error(f"Agent init failed — check API key. Details: {e}")
            st.stop()

        st.write("Running preprocessing + default segmentation...")
        try:
            agent.run_default_pipeline(method="kmeans")
        except Exception as e:
            status.update(label="Pipeline failed", state="error")
            st.error(f"Segmentation pipeline failed: {e}")
            st.stop()

        for step in agent.execution_log:
            st.write(f"✅ {step}")
        status.update(label="Agent ready", state="complete")

    st.session_state.agent = agent

agent = st.session_state.agent

# ── KPI Dashboard ─────────────────────────────────────────────────────
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

# ── Tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💬 Agent Chat",
    "📊 EDA Summary",
    "🎯 Customer Segments",
    "📈 Model Evaluation",
    "💡 Insights & Retention",
    "📋 Recommendations",
])

# Tab 1: Agent Chat
with tab1:
    st.subheader("Ask the Banking Analytics Agent")
    st.caption(
        "Try: \"Segment customers using kmeans with 4 clusters\", "
        "\"What insights can you derive about each segment?\", "
        "\"Which customers are at risk of churning?\", "
        "\"Show retention strategies for Priority customers\", "
        "\"Which features drive the segmentation?\", "
        "\"Show only the priority ones\" (after segmenting)"
    )
    user_query = st.text_input(
        "Type your query:",
        placeholder="e.g., What insights can you derive about each segment?",
    )

    if st.button("Send Query") and user_query:
        with st.spinner("Agent processing..."):
            result = agent.process_query(user_query)

        st.markdown("### Agent Response")
        st.write(result["answer"])

        with st.expander("🔍 Agent reasoning / execution log"):
            for step in result["log"]:
                st.text(f"• {step}")

        st.download_button(
            "📥 Download response as report (.txt)",
            data=f"Query: {user_query}\n\nAnswer:\n{result['answer']}\n\n"
                 f"Execution log:\n" + "\n".join(result["log"]),
            file_name="agent_response_report.txt",
            mime="text/plain",
        )

# Tab 2: EDA Summary (NARRATIVE — not raw stats)
with tab2:
    st.subheader("📊 Executive EDA Summary")
    st.markdown(generate_eda_summary(agent.raw_df))

    with st.expander("🔍 Raw EDA Details (for analysts)"):
        eda_raw = run_eda(agent.raw_df)
        col1, col2 = st.columns(2)
        col1.metric("Total Records", eda_raw["total_rows"])
        col2.metric("Total Features", eda_raw["total_columns"])
        st.write("#### Missing Values", eda_raw["missing_values"])
        st.dataframe(pd.DataFrame(eda_raw["numeric_summary"]).T)

# Tab 3: Customer Segments
with tab3:
    st.subheader("Customer Segments & Export")
    seg_df = agent.segmented_df

    if seg_df is not None:
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button(
                "📥 Download Segmented Customers (CSV)",
                data=seg_df.to_csv(index=False).encode("utf-8"),
                file_name="segmented_customers.csv",
                mime="text/csv",
            )
        with dl2:
            kpi_json = json.dumps(compute_business_kpis(seg_df), indent=2)
            st.download_button(
                "📥 Download KPI Summary (JSON)",
                data=kpi_json,
                file_name="segment_kpi_summary.json",
                mime="application/json",
            )
        with dl3:
            personas = generate_customer_personas(seg_df)
            st.download_button(
                "📥 Download Personas (JSON)",
                data=json.dumps(personas, indent=2),
                file_name="customer_personas.json",
                mime="application/json",
            )

        # Visualizations
        st.write("#### Segment Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Count plot
        seg_counts = seg_df["Segment"].value_counts()
        colors = {"Priority": "#2ecc71", "Regular": "#3498db", "Dormant": "#e74c3c"}
        bar_colors = [colors.get(s, "#95a5a6") for s in seg_counts.index]
        axes[0].bar(seg_counts.index, seg_counts.values, color=bar_colors)
        axes[0].set_title("Customer Count by Segment")
        axes[0].set_ylabel("Number of Customers")

        # Balance comparison
        bal_col = "avg_balance" if "avg_balance" in seg_df.columns else "current_balance"
        if bal_col in seg_df.columns:
            seg_avg_bal = seg_df.groupby("Segment")[bal_col].mean()
            bar_colors2 = [colors.get(s, "#95a5a6") for s in seg_avg_bal.index]
            axes[1].bar(seg_avg_bal.index, seg_avg_bal.values, color=bar_colors2)
            axes[1].set_title(f"Average {bal_col.replace('_', ' ').title()} by Segment")
            axes[1].set_ylabel("₹ Amount")

        plt.tight_layout()
        st.pyplot(fig)

        # Feature importance
        if st.checkbox("Show Feature Importance"):
            with st.spinner("Computing feature importance..."):
                importance = select_most_important_features(seg_df)
                if "feature_importance" in importance:
                    imp_df = pd.DataFrame(importance["feature_importance"])
                    fig2, ax = plt.subplots(figsize=(8, 4))
                    sns.barplot(data=imp_df, x="importance", y="feature", ax=ax, palette="Blues_r")
                    ax.set_title("Most Important Features for Customer Segmentation")
                    st.pyplot(fig2)
                    st.caption(importance.get("interpretation", ""))

        st.dataframe(seg_df.head(100))
    else:
        st.info("No segmentation has been run yet.")

# Tab 4: Model Evaluation
with tab4:
    st.subheader("Clustering Model Performance")
    eval_data = agent.evaluation_metrics

    if eval_data:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Method", str(eval_data.get("method", "N/A")).upper())
        col2.metric("Silhouette", eval_data.get("silhouette_score", "N/A"))
        col3.metric("Davies-Bouldin", eval_data.get("davies_bouldin_score", "N/A"))
        col4.metric("Calinski-Harabasz", eval_data.get("calinski_harabasz_score", "N/A"))

        st.info(f"**Quality:** {eval_data.get('quality', 'N/A')}")

        if "segment_distribution" in eval_data:
            st.write("#### Segment Distribution")
            st.json(eval_data["segment_distribution"])

        if st.button("🔬 Compare KMeans vs Hierarchical vs DBSCAN"):
            with st.spinner("Running all methods..."):
                from src.tools.segmentation_tool import compare_clustering_methods
                comparison = compare_clustering_methods(agent.processed_df)
            st.json(comparison)
    else:
        st.write("Run segmentation to display evaluation metrics.")

# Tab 5: Insights & Retention
with tab5:
    st.subheader("💡 Segment Insights & Retention Strategies")

    if agent.segmented_df is not None:
        sub1, sub2 = st.tabs(["📊 Data-Driven Insights", "🛡️ Retention Strategies"])

        with sub1:
            st.markdown(generate_segment_insights(agent.segmented_df))

            with st.expander("👤 Customer Personas"):
                personas = generate_customer_personas(agent.segmented_df)
                for seg, data in personas.items():
                    if isinstance(data, dict):
                        st.markdown(f"**{data.get('persona_name', seg)}**")
                        st.markdown(data.get("behavioral_profile", ""))
                        st.markdown(data.get("actionable_insight", ""))
                        st.markdown("---")

        with sub2:
            st.markdown(get_retention_strategies(agent.segmented_df))

            with st.expander("⚠️ At-Risk Customer Detection"):
                from src.tools.explainability_tool import identify_at_risk_customers
                at_risk = identify_at_risk_customers(agent.segmented_df)
                st.markdown(at_risk.get("summary", "No at-risk customers detected."))
    else:
        st.info("Run segmentation to view insights and retention strategies.")

# Tab 6: Recommendations
with tab6:
    st.subheader("💳 Cross-Selling & Personalization Engine")
    selected_segment = st.selectbox("Select Segment:", ["Priority", "Regular", "Dormant"])

    if st.button("Generate Strategy"):
        strategy = get_cross_sell_recommendations(selected_segment)
        st.success(f"Strategy for **{selected_segment}**:")
        st.markdown(strategy)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🥇 Priority\n- Wealth Management\n- Premium Travel Cards\n- Dedicated RM")
    with col2:
        st.markdown("### 🥈 Regular\n- Personal Loans\n- Auto Loans\n- Shopping Rewards")
    with col3:
        st.markdown("### 💤 Dormant\n- UPI Cashback\n- Zero-Balance Account\n- Win-Back Campaigns")