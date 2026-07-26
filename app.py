import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from src.agent import RetailBankingAgent
from src.tools.eda_tool import run_eda, validate_dataset, generate_eda_summary
from src.tools.explainability_tool import (
    get_cross_sell_recommendations,
    generate_customer_personas,
    generate_segment_insights,
    get_retention_strategies,
    identify_at_risk_customers,
)
from src.tools.kpi_tool import compute_business_kpis
from src.tools.feature_engineering import select_most_important_features
from src.config import DEFAULT_SAMPLE_SIZE
from src.logger_setup import get_logger

logger = get_logger("app")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Retail Banking AI Agent | Segmentation & Personalization",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — FIXED COLORS (NO CSS VARIABLES)
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #3B4A63;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', 'Inter', sans-serif !important;
    }

    .stApp { background: #969799; }
    .main > div { padding: 1rem 2.2rem 2rem 2.2rem; }

    /* ── Headings ─────────────────────────────────────────────────── */
    h1 {
        font-size: 2.1rem !important;
        font-weight: 700 !important;
        color: #0F2038 !important;
        -webkit-text-fill-color: #0F2038 !important;
        letter-spacing: -0.5px;
        margin-bottom: 0 !important;
    }
    h2 {
        font-weight: 600 !important;
        color: #0F2038 !important;
        font-size: 1.4rem !important;
        letter-spacing: -0.2px;
    }
    h3 {
        font-weight: 600 !important;
        color: #163158 !important;
        font-size: 1.15rem !important;
    }
    h4 { font-weight: 600 !important; color: #163158 !important; }

    /* ── Metric cards ─────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 18px 16px;
        border: 1px solid #E3E6EC;
        border-top: 3px solid #C9A227;
        box-shadow: 0 1px 2px rgba(16, 35, 63, 0.04);
    }
    [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #0F2038 !important;
        -webkit-text-fill-color: #0F2038 !important;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600 !important;
        color: #6B7789 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        color: #6B7789 !important;
    }

    /* ── Buttons ──────────────────────────────────────────────────── */
    .stButton > button {
        background: #0F2038;
        color: #FFFFFF !important;
        border-radius: 8px;
        padding: 10px 22px;
        font-weight: 600;
        font-size: 0.9rem;
        border: 1px solid #0F2038;
    }
    .stButton > button:hover {
        background: #163158;
        border-color: #163158;
    }
    .stButton > button[kind="primary"] {
        background: #C9A227;
        border-color: #C9A227;
        color: #0A1628 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #DDBE55;
        border-color: #DDBE55;
    }
    .stButton > button[kind="secondary"] {
        background: #FFFFFF;
        color: #0F2038 !important;
        border: 1.5px solid #E3E6EC;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #163158;
    }

    /* ── Inputs ───────────────────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border: 1.5px solid #E3E6EC;
        border-radius: 10px;
        padding: 11px 14px;
        font-size: 0.95rem;
        background: #FFFFFF;
        color: #10233F !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #163158;
        box-shadow: 0 0 0 3px rgba(22, 49, 88, 0.12);
        outline: none;
    }
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: #8592A6 !important;
    }

    /* ── Sidebar ──────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #0A1628;
        border-right: none;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #C7CEDB !important;
    }
    [data-testid="stSidebar"] .stTextInput > div > div > input {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.16);
        color: #FFFFFF !important;
        border-radius: 8px;
    }
    [data-testid="stSidebar"] .stTextInput > div > div > input::placeholder {
        color: #8592A6 !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.08);
        color: #FFFFFF !important;
        border: 1px solid rgba(255,255,255,0.2);
        width: 100%;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.16);
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-top: 3px solid #C9A227;
        padding: 12px 10px;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #DDBE55 !important;
        -webkit-text-fill-color: #DDBE55 !important;
        font-size: 1.25rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #8FA0BC !important;
    }
    [data-testid="stSidebar"] .stAlert {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.14);
    }
    [data-testid="stSidebar"] .stAlert p {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stCheckbox label span {
        color: #C7CEDB !important;
    }
    [data-testid="stSidebar"] hr {
        background: rgba(255,255,255,0.12);
    }

    /* ── Tabs ─────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: #EAECF1;
        padding: 4px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 9px 18px;
        font-weight: 600;
        font-size: 0.87rem;
        color: #6B7789 !important;
    }
    .stTabs [aria-selected="true"] {
        background: #0F2038;
        color: #FFFFFF !important;
    }
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background: rgba(16, 35, 63, 0.06);
        color: #0F2038 !important;
    }

    /* ── Expander ─────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #163158 !important;
        background: #EAF1F7;
        border-radius: 8px;
        padding: 10px 14px !important;
        border: 1px solid #E3E6EC !important;
    }

    /* ── Dataframe ────────────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        border: 1px solid #E3E6EC;
    }

    /* ── Alerts ───────────────────────────────────────────────────── */
    .stAlert { border-radius: 10px; border-left: 4px solid; }
    .stAlert p, .stAlert div { color: #3B4A63 !important; }
    .stSuccess { background: #EAF1F7; }
    .stWarning { background: #FBF4DE; }
    .stInfo { background: #EEF0F3; }

    /* ── Agent response ───────────────────────────────────────────── */
    .agent-response {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 20px 22px;
        border: 1px solid #E3E6EC;
        border-left: 4px solid #C9A227;
        margin: 10px 0;
        line-height: 1.65;
    }
    .agent-response p,
    .agent-response li,
    .agent-response span,
    .agent-response div,
    .agent-response h2,
    .agent-response h3,
    .agent-response h4 {
        color: #10233F !important;
    }

    /* ── KPI section ──────────────────────────────────────────────── */
    .kpi-section {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 22px 24px;
        margin: 14px 0;
        border: 1px solid #E3E6EC;
    }

    /* ── File uploader ────────────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        border-radius: 12px;
        border: 1.5px dashed #A9B4C4;
        padding: 18px;
        background: #FFFFFF;
    }

    hr {
        border: none;
        height: 1px;
        background: #E3E6EC;
        margin: 1.4rem 0;
    }

    /* ── Feature cards ────────────────────────────────────────────── */
    .info-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 22px 20px;
        border: 1px solid #E3E6EC;
        border-top: 3px solid #C9A227;
        height: 100%;
    }
    .info-card h4 { margin: 10px 0 6px 0; color: #0F2038 !important; }
    .info-card p { color: #6B7789; font-size: 0.88rem; margin: 0; }

    /* ── Segment cards ────────────────────────────────────────────── */
    .segment-card {
        border-radius: 14px;
        padding: 24px;
        height: 100%;
    }
    .segment-card h3 { color: #FFFFFF !important; margin-top: 4px; }
    .segment-card .sub { opacity: 0.85; font-size: 0.85rem; margin-bottom: 12px; color: #FFFFFF !important; }
    .segment-card ul { list-style: none; padding: 0; margin: 0; }
    .segment-card li {
        padding: 5px 0;
        font-size: 0.9rem;
        border-top: 1px solid rgba(255,255,255,0.15);
        color: #FFFFFF !important;
    }
    .segment-card li:first-child { border-top: none; }

    /* ── Spinner ──────────────────────────────────────────────────── */
    .stSpinner > div { border-color: #0F2038 !important; }

    /* ── Caption ──────────────────────────────────────────────────── */
    .stCaption { color: #6B7789 !important; }

    /* ── Select box ───────────────────────────────────────────────── */
    .stSelectbox > div > div { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# Consistent segment → color mapping
SEGMENT_COLORS = {"Priority": "#C9A227", "Regular": "#4C7EA8", "Dormant": "#8592A6"}
SEGMENT_EMOJI = {"Priority": "🥇", "Regular": "🥈", "Dormant": "💤"}

# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════
col1, col2, col3 = st.columns([1, 8, 1])
with col1:
    st.markdown("<div style='font-size: 3rem; text-align: center;'>🏦</div>", unsafe_allow_html=True)
with col2:
    st.title("Retail Banking AI Agent")
    st.markdown(
        "<p style='font-size: 1rem; color: var(--ink-500); margin-top: -6px; font-weight: 500;'>"
        "Customer Segmentation &amp; Personalization Engine — Powered by Gemini"
        "</p>",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        "<div style='background: var(--navy-900); color: #FFFFFF; "
        "border-radius: 50px; padding: 8px 14px; text-align: center; font-size: 0.78rem; "
        "font-weight: 700; margin-top: 10px; border: 1px solid var(--gold-500);'>v2.0</div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    api_key = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        placeholder="Enter your API key...",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📂 Upload Customer Dataset",
        type=["csv"],
    )

    st.markdown("---")
    st.markdown("### ⚡ Performance")

    use_full_dataset = st.checkbox(
        "🔓 Use full dataset",
        value=False,
        help="May be slow for large files"
    )

    sample_size = st.number_input(
        "Sample size (rows)",
        min_value=1000,
        max_value=1_000_000,
        value=DEFAULT_SAMPLE_SIZE,
        step=10000,
    )

    st.markdown("---")
    st.markdown("### 📊 System Status")

    if "agent" in st.session_state:
        st.success("✅ Agent Active")
        if st.session_state.agent.segmented_df is not None:
            seg_df = st.session_state.agent.segmented_df
            total = len(seg_df)
            priority = (seg_df["Segment"] == "Priority").sum()
            regular = (seg_df["Segment"] == "Regular").sum()
            dormant = (seg_df["Segment"] == "Dormant").sum()

            st.metric("Total Customers", f"{total:,}")

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("🥇 Priority", f"{priority:,}")
            col_b.metric("🥈 Regular", f"{regular:,}")
            col_c.metric("💤 Dormant", f"{dormant:,}")
    else:
        st.warning("⏳ Awaiting data...")

    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; font-size: 0.78rem; color: #8FA0BC;'>"
        "Built with Streamlit + Google Gemini<br>"
        "© 2026 Banking Analytics Agent</p>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file is None:
    # Welcome screen
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.markdown(
            "<div style='text-align: center; padding: 30px 20px;'>"
            "<div style='font-size: 3.2rem;'>📊</div>"
            "<h2 style='color: var(--navy-900); margin-top: 16px;'>Ready to Analyze Your Customers</h2>"
            "<p style='color: var(--ink-500); font-size: 1.02rem; max-width: 520px; margin: 16px auto;'>"
            "Upload your bank transaction dataset and let the AI agent automatically "
            "segment customers, generate insights, and recommend retention strategies — "
            "all through natural language queries.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    features = [
        ("🔍", "Automated EDA", "Understand your data in seconds with AI-powered analysis"),
        ("🎯", "Smart Segmentation", "4 clustering methods with automatic model evaluation"),
        ("💡", "Actionable Insights", "7 types of data-driven discoveries about your segments"),
        ("🛡️", "Retention Engine", "Identify at-risk customers and prevent churn"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(
                f"<div class='info-card'>"
                f"<div style='font-size: 2.1rem;'>{icon}</div>"
                f"<h4>{title}</h4><p>{desc}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.stop()

# Load and validate data
with st.spinner("📥 Loading and validating dataset..."):
    try:
        df_raw = load_data(uploaded_file)
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        st.stop()

validation_issues = validate_dataset(df_raw)
if validation_issues:
    for issue in validation_issues:
        st.sidebar.warning(f"⚠️ {issue}")
    if df_raw.empty:
        st.error("❌ Dataset is empty.")
        st.stop()

original_row_count = len(df_raw)
if not use_full_dataset and original_row_count > sample_size:
    df_raw = df_raw.sample(n=int(sample_size), random_state=42).reset_index(drop=True)
    st.sidebar.success(f"📊 {original_row_count:,} → {len(df_raw):,} rows")
else:
    st.sidebar.success(f"📊 {len(df_raw):,} records loaded")

if api_key:
    os.environ["GEMINI_API_KEY"] = api_key

# ═══════════════════════════════════════════════════════════════════════════
# AGENT INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════
if "agent" not in st.session_state or st.sidebar.button("🔄 Re-initialize Agent"):
    progress_bar = st.progress(0, text="Initializing agent...")

    try:
        progress_bar.progress(20, text="Creating Retail Banking Agent...")
        agent = RetailBankingAgent(df_raw)

        progress_bar.progress(50, text="Running preprocessing pipeline...")
        agent.run_default_pipeline(method="kmeans")

        progress_bar.progress(90, text="Computing KPIs...")

        progress_bar.progress(100, text="✅ Agent ready!")
        time.sleep(0.5)
        progress_bar.empty()

        st.session_state.agent = agent

    except Exception as e:
        progress_bar.empty()
        st.error(f"❌ Agent initialization failed: {e}")
        st.stop()

agent = st.session_state.agent

# ═══════════════════════════════════════════════════════════════════════════
# KPI DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="kpi-section">', unsafe_allow_html=True)
st.markdown("### 📌 Business KPI Dashboard")

kpis = compute_business_kpis(agent.segmented_df) if agent.segmented_df is not None else {}

if kpis:
    total = kpis.get('total_customers', 1)

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:
        st.metric("👥 Total Customers", f"{kpis.get('total_customers', 0):,}")
    with k2:
        priority_pct = kpis.get('priority_customers', 0) / total * 100
        st.metric("🥇 Priority", f"{kpis.get('priority_customers', 0):,}", delta=f"{priority_pct:.1f}% of base")
    with k3:
        regular_pct = kpis.get('regular_customers', 0) / total * 100
        st.metric("🥈 Regular", f"{kpis.get('regular_customers', 0):,}", delta=f"{regular_pct:.1f}% of base")
    with k4:
        dormant_pct = kpis.get('dormant_customers', 0) / total * 100
        st.metric("💤 Dormant", f"{kpis.get('dormant_customers', 0):,}", delta=f"{dormant_pct:.1f}% of base")
    with k5:
        st.metric("💰 Avg Balance", f"₹{kpis.get('avg_balance', 0):,.0f}")
    with k6:
        st.metric("🎯 Cross-Sell Potential", f"{kpis.get('cross_sell_potential', 0):,}", delta="upgrade candidates")
else:
    st.info("📊 KPIs will appear once segmentation completes.")

st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💬 Agent Chat",
    "📊 EDA Summary",
    "🎯 Customer Segments",
    "📈 Model Evaluation",
    "💡 Insights & Retention",
    "📋 Recommendations",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: AGENT CHAT
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("💬 Ask the Banking Analytics Agent")
        st.caption(
            "Ask anything about your customers — the agent will automatically select "
            "the right tools to answer your query."
        )

        with st.expander("💡 Suggested Queries"):
            suggestions = [
                "Segment customers using kmeans with 4 clusters",
                "What insights can you derive about each segment?",
                "Which customers are at risk of churning?",
                "Show retention strategies for Priority customers",
                "Which features drive the segmentation?",
                "On what basis were priority customers selected?",
                "Which regular customers can be converted to priority?",
                "Show only the priority ones",
                "Compare clustering methods",
                "Identify any edge cases in the segmentation",
            ]
            cols = st.columns(2)
            for i, sug in enumerate(suggestions):
                with cols[i % 2]:
                    if st.button(sug, key=f"sug_{i}", use_container_width=True):
                        st.session_state.pending_query = sug

        user_query = st.text_area(
            "Type your query:",
            placeholder="e.g., What insights can you derive about each segment?",
            height=80,
            key="chat_input",
        )

        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            send_btn = st.button("🚀 Send Query", type="primary", use_container_width=True)

        if (send_btn and user_query) or "pending_query" in st.session_state:
            query_to_process = user_query if (send_btn and user_query) else st.session_state.get("pending_query", "")
            if "pending_query" in st.session_state:
                del st.session_state.pending_query

            with st.spinner("🤖 Agent is analyzing your query and selecting tools..."):
                result = agent.process_query(query_to_process)

            st.markdown("### 📝 Agent Response")
            st.markdown(f'<div class="agent-response">{result["answer"]}</div>', unsafe_allow_html=True)

            with st.expander("🔍 View Agent Reasoning & Execution Log"):
                for i, step in enumerate(result["log"]):
                    icon = "🎯" if "Tool:" in step else "📋"
                    st.text(f"{icon} {step}")

            st.download_button(
                "📥 Download Response Report",
                data=f"QUERY: {query_to_process}\n\nRESPONSE:\n{result['answer']}\n\n"
                     f"EXECUTION LOG:\n" + "\n".join(result["log"]),
                file_name="agent_response.txt",
                mime="text/plain",
            )

    with col_right:
        st.subheader("📊 Quick Stats")

        if agent.segmented_df is not None:
            seg_counts = agent.segmented_df["Segment"].value_counts()
            fig_pie = go.Figure(data=[
                go.Pie(
                    labels=seg_counts.index,
                    values=seg_counts.values,
                    hole=0.55,
                    marker=dict(
                        colors=[SEGMENT_COLORS.get(s, "#95a5a6") for s in seg_counts.index],
                        line=dict(color="white", width=2),
                    ),
                )
            ])
            fig_pie.update_layout(
                title="Segment Distribution",
                height=300,
                margin=dict(t=40, b=0, l=0, r=0),
                showlegend=True,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color="#3B4A63"),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("---")
            st.markdown("**Segment Counts**")
            for seg, count in seg_counts.items():
                pct = count / len(agent.segmented_df) * 100
                emoji = SEGMENT_EMOJI.get(seg, "📌")
                st.metric(f"{emoji} {seg}", f"{count:,}", delta=f"{pct:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: EDA SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Executive EDA Summary")

    eda_summary_text = generate_eda_summary(agent.raw_df)
    st.markdown(f'<div class="agent-response">{eda_summary_text}</div>', unsafe_allow_html=True)

    with st.expander("🔍 Technical EDA Details"):
        eda_raw = run_eda(agent.raw_df)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{eda_raw['total_rows']:,}")
        col2.metric("Total Features", eda_raw["total_columns"])
        col3.metric("Numeric Columns", len(eda_raw["numeric_summary"]))
        col4.metric("Missing Values", sum(eda_raw["missing_values"].values()))

        st.write("#### Missing Values Summary")
        missing_df = pd.DataFrame(
            list(eda_raw["missing_values"].items()),
            columns=["Column", "Missing Count"],
        )
        st.dataframe(missing_df, use_container_width=True)

        st.write("#### Numeric Summary Statistics")
        st.dataframe(pd.DataFrame(eda_raw["numeric_summary"]).T, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: CUSTOMER SEGMENTS
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🎯 Customer Segments & Data Export")
    seg_df = agent.segmented_df

    if seg_df is not None:
        dl1, dl2, dl3, dl4 = st.columns(4)
        with dl1:
            st.download_button(
                "📥 Segmented CSV",
                data=seg_df.to_csv(index=False).encode("utf-8"),
                file_name="segmented_customers.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl2:
            kpi_json = json.dumps(compute_business_kpis(seg_df), indent=2)
            st.download_button(
                "📥 KPI Summary (JSON)",
                data=kpi_json,
                file_name="kpi_summary.json",
                mime="application/json",
                use_container_width=True,
            )
        with dl3:
            personas = generate_customer_personas(seg_df)
            st.download_button(
                "📥 Personas (JSON)",
                data=json.dumps(personas, indent=2),
                file_name="customer_personas.json",
                mime="application/json",
                use_container_width=True,
            )
        with dl4:
            st.download_button(
                "📥 Full Report (JSON)",
                data=json.dumps({
                    "kpis": compute_business_kpis(seg_df),
                    "personas": personas,
                    "segments": seg_df["Segment"].value_counts().to_dict(),
                }, indent=2),
                file_name="full_report.json",
                mime="application/json",
                use_container_width=True,
            )

        st.markdown("---")

        col_v1, col_v2 = st.columns(2)

        with col_v1:
            st.write("#### Segment Distribution")
            seg_counts = seg_df["Segment"].value_counts()

            fig1 = go.Figure(data=[
                go.Bar(
                    x=seg_counts.index,
                    y=seg_counts.values,
                    marker_color=[SEGMENT_COLORS.get(s, "#95a5a6") for s in seg_counts.index],
                    text=seg_counts.values,
                    textposition="outside",
                )
            ])
            fig1.update_layout(
                height=350,
                margin=dict(t=20, b=0, l=0, r=0),
                yaxis_title="Number of Customers",
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color="#3B4A63"),
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col_v2:
            st.write("#### Average Balance by Segment")
            bal_col = "avg_balance" if "avg_balance" in seg_df.columns else "current_balance"
            if bal_col in seg_df.columns:
                seg_avg_bal = seg_df.groupby("Segment")[bal_col].mean()

                fig2 = go.Figure(data=[
                    go.Bar(
                        x=seg_avg_bal.index,
                        y=seg_avg_bal.values,
                        marker_color=[SEGMENT_COLORS.get(s, "#95a5a6") for s in seg_avg_bal.index],
                        text=[f"₹{v:,.0f}" for v in seg_avg_bal.values],
                        textposition="outside",
                    )
                ])
                fig2.update_layout(
                    height=350,
                    margin=dict(t=20, b=0, l=0, r=0),
                    yaxis_title="₹ Amount",
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color="#3B4A63"),
                )
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        if st.checkbox("🔬 Show Feature Importance Analysis"):
            with st.spinner("Computing Random Forest feature importance..."):
                importance = select_most_important_features(seg_df)
                if "feature_importance" in importance:
                    imp_df = pd.DataFrame(importance["feature_importance"])

                    fig3 = go.Figure(data=[
                        go.Bar(
                            y=imp_df["feature"],
                            x=imp_df["importance"],
                            orientation="h",
                            marker=dict(
                                color=imp_df["importance"],
                                colorscale=[[0, "#EAF1F7"], [1, "#0F2038"]],
                                showscale=True,
                                colorbar=dict(title="Importance"),
                            ),
                        )
                    ])
                    fig3.update_layout(
                        height=400,
                        title="Most Definitive Features for Customer Segmentation",
                        margin=dict(t=40, b=0, l=0, r=0),
                        yaxis=dict(autorange="reversed"),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, sans-serif", color="#3B4A63"),
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                    st.info(f"💡 {importance.get('interpretation', '')}")

        st.markdown("---")
        st.write("#### Segmented Customer Data")
        st.dataframe(seg_df.head(100), use_container_width=True)
        st.caption(f"Showing 100 of {len(seg_df):,} customers")
    else:
        st.info("No segmentation has been run yet. Ask the agent to segment customers.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📈 Clustering Model Performance")
    eval_data = agent.evaluation_metrics

    if eval_data:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            method = str(eval_data.get("method", "N/A")).upper()
            st.metric("📊 Method", method)
        with col2:
            st.metric("🎯 Silhouette Score", eval_data.get("silhouette_score", "N/A"))
        with col3:
            st.metric("📉 Davies-Bouldin", eval_data.get("davies_bouldin_score", "N/A"))
        with col4:
            st.metric("📈 Calinski-Harabasz", eval_data.get("calinski_harabasz_score", "N/A"))
        
        quality = eval_data.get("quality", "N/A")
        if "Strong" in str(quality):
            st.success(f"✅ **Model Quality:** {quality}")
        elif "Moderate" in str(quality):
            st.warning(f"⚠️ **Model Quality:** {quality}")
        else:
            st.info(f"📊 **Model Quality:** {quality}")
        
        st.markdown("---")
        if "segment_distribution" in eval_data:
            st.write("#### Segment Distribution")
            st.json(eval_data["segment_distribution"])
        
        st.markdown("---")
        col_comp1, col_comp2 = st.columns([1, 3])
        with col_comp1:
            if st.button("🔬 Compare All Methods", type="primary", use_container_width=True):
                with st.spinner("Running KMeans, Hierarchical, and DBSCAN..."):
                    from src.tools.segmentation_tool import compare_clustering_methods
                    comparison = compare_clustering_methods(agent.processed_df)
                
                st.write("#### Side-by-Side Method Comparison")
                
                # Create comparison table
                comp_data = []
                for meth, metrics in comparison.items():
                    comp_data.append({
                        "Method": meth.upper(),
                        "Silhouette": metrics.get("silhouette_score", "N/A"),
                        "Davies-Bouldin": metrics.get("davies_bouldin_score", "N/A"),
                        "Calinski-Harabasz": metrics.get("calinski_harabasz_score", "N/A"),
                        "Quality": metrics.get("quality", "N/A"),
                    })
                
                comp_df = pd.DataFrame(comp_data)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
                
                # Find best method safely
                try:
                    # Convert silhouette column to numeric, coercing errors to NaN
                    sil_scores = pd.to_numeric(comp_df["Silhouette"], errors='coerce')
                    # Drop NaN values
                    valid_scores = sil_scores.dropna()
                    
                    if len(valid_scores) > 0:
                        best_idx = valid_scores.idxmax()
                        best_method = comp_df.loc[best_idx, "Method"]
                        best_score = comp_df.loc[best_idx, "Silhouette"]
                        st.success(f"🏆 **Best Method:** {best_method} (Silhouette: {best_score})")
                    else:
                        st.info("⚠️ Could not determine best method — no valid silhouette scores available.")
                except Exception:
                    st.info("⚠️ Could not compare methods automatically. Please review the table above.")
    else:
        st.info("📊 Run segmentation to display evaluation metrics.")
# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: INSIGHTS & RETENTION
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("💡 Segment Insights & Retention Strategies")

    if agent.segmented_df is not None:
        sub1, sub2, sub3 = st.tabs([
            "📊 Data-Driven Insights",
            "🛡️ Retention Strategies",
            "⚠️ At-Risk Customers",
        ])

        with sub1:
            insights_text = generate_segment_insights(agent.segmented_df)
            st.markdown(f'<div class="agent-response">{insights_text}</div>', unsafe_allow_html=True)

            st.markdown("---")

            with st.expander("👤 View Customer Personas"):
                personas = generate_customer_personas(agent.segmented_df)
                for seg, data in personas.items():
                    if isinstance(data, dict):
                        emoji = SEGMENT_EMOJI.get(seg, "📌")
                        st.markdown(f"### {emoji} {data.get('persona_name', seg)}")
                        st.markdown(data.get("behavioral_profile", ""))
                        st.markdown(f"**Strategy:** {data.get('actionable_insight', '')}")
                        st.markdown("---")

        with sub2:
            retention_text = get_retention_strategies(agent.segmented_df)
            st.markdown(f'<div class="agent-response">{retention_text}</div>', unsafe_allow_html=True)

        with sub3:
            st.markdown("### ⚠️ At-Risk Customer Detection")
            st.caption("Customers showing early warning signs of churn or downgrade")

            at_risk = identify_at_risk_customers(agent.segmented_df)

            if "summary" in at_risk:
                st.markdown(at_risk["summary"])

            if at_risk.get("priority_downgrade_risk"):
                with st.expander(f"🔴 Priority Downgrade Risk ({len(at_risk['priority_downgrade_risk'])} customers)"):
                    st.dataframe(pd.DataFrame(at_risk["priority_downgrade_risk"]), use_container_width=True)

            if at_risk.get("regular_dormancy_risk"):
                with st.expander(f"🟡 Regular → Dormant Risk ({len(at_risk['regular_dormancy_risk'])} customers)"):
                    st.dataframe(pd.DataFrame(at_risk["regular_dormancy_risk"]), use_container_width=True)

            if at_risk.get("high_value_dormant_churn"):
                with st.expander(f"🟠 High-Value Dormant Churn Risk ({len(at_risk['high_value_dormant_churn'])} customers)"):
                    st.dataframe(pd.DataFrame(at_risk["high_value_dormant_churn"]), use_container_width=True)
    else:
        st.info("Run segmentation to view insights and retention strategies.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 6: RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("💳 Cross-Selling & Personalization Engine")
    st.caption("Generate tailored product recommendations for each customer segment")

    col_rec1, col_rec2 = st.columns([1, 2])

    with col_rec1:
        selected_segment = st.selectbox("Select Customer Segment:", ["Priority", "Regular", "Dormant"])

        if st.button("🎯 Generate Strategy", type="primary", use_container_width=True):
            strategy = get_cross_sell_recommendations(selected_segment)
            st.session_state.rec_strategy = strategy
            st.session_state.rec_segment = selected_segment

    with col_rec2:
        if "rec_strategy" in st.session_state:
            st.success(f"📋 Strategy for **{st.session_state.rec_segment}** Segment")
            st.markdown(f'<div class="agent-response">{st.session_state.rec_strategy}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.write("### Segment Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"<div class='segment-card' style='background: linear-gradient(160deg, {SEGMENT_COLORS['Priority']}, #8A6D14);'>"
            "<h3>🥇 Priority</h3><p class='sub'>High-Net-Worth Customers</p>"
            "<ul><li>💰 Wealth Management</li><li>✈️ Premium Travel Cards</li>"
            "<li>👨‍💼 Dedicated RM</li><li>🏆 Tiered Fixed Deposits</li></ul></div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"<div class='segment-card' style='background: linear-gradient(160deg, {SEGMENT_COLORS['Regular']}, #2E5A80);'>"
            "<h3>🥈 Regular</h3><p class='sub'>Daily Banking Users</p>"
            "<ul><li>💳 Personal Loans</li><li>🚗 Auto Loans</li>"
            "<li>🛍️ Shopping Rewards</li><li>📱 Digital Banking</li></ul></div>",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"<div class='segment-card' style='background: linear-gradient(160deg, {SEGMENT_COLORS['Dormant']}, #566072);'>"
            "<h3>💤 Dormant</h3><p class='sub'>Inactive Accounts</p>"
            "<ul><li>📱 UPI Cashback</li><li>🏦 Zero-Balance A/C</li>"
            "<li>📧 Win-Back Campaigns</li><li>💵 High-Yield Savings</li></ul></div>",
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: var(--ink-500); font-size: 0.82rem;'>"
    "🏦 Retail Banking AI Agent | Built with Streamlit + Google Gemini | "
    "Customer Segmentation &amp; Personalization Engine v2.0"
    "</p>",
    unsafe_allow_html=True,
)