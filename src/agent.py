import re
import pandas as pd
from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, MODEL_NAME
from src.logger_setup import get_logger
from src.tools.eda_tool import run_eda, query_eda_metrics, generate_eda_summary
from src.tools.feature_engineering import (
    preprocess_and_aggregate_customer_data,
    select_most_important_features,
)
from src.tools.segmentation_tool import (
    segment_customers,
    find_upgrade_candidates,
    compare_clustering_methods,
    identify_edge_cases,
)
from src.tools.explainability_tool import (
    get_segment_profiles,
    get_cross_sell_recommendations,
    explain_customer_segment,
    generate_customer_personas,
    generate_segment_insights,
    identify_at_risk_customers,
    get_retention_strategies,
)
from src.tools.kpi_tool import compute_business_kpis

logger = get_logger(__name__)

_VAGUE_QUALIFIER = re.compile(r"\b(best|top|good|great|ideal|focus|target|worth|important|key)\b", re.IGNORECASE)
_METRIC_HINTS = [
    "balance", "frequency", "spend", "transaction", "priority", "regular",
    "dormant", "amount", "count", "cluster", "kmeans", "recent", "recency",
    "trend", "segment", "segmentation",
]
_SEGMENT_KEYWORDS = [
    "balance", "frequency", "kmeans", "cluster", "rule", "dbscan",
    "hierarchical", "spend", "method", "using", "with",
]


class RetailBankingAgent:
    def __init__(self, raw_df: pd.DataFrame):
        self.raw_df = raw_df
        self.processed_df = None
        self.segmented_df = None
        self.evaluation_metrics = {}
        self.execution_log = []
        self.last_filtered_df = None
        self.last_filter_description = None

        self.client = (
            genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else genai.Client()
        )

        def _log(step: str):
            self.execution_log.append(step)
            logger.info(step)

        def _find_customer_row(customer_id: str):
            df_to_search = (
                self.segmented_df if self.segmented_df is not None else self.raw_df
            )
            id_col = next(
                (c for c in df_to_search.columns if "cust" in c.lower() or "id" in c.lower()),
                df_to_search.columns[0],
            )
            matched = df_to_search[
                df_to_search[id_col].astype(str).str.strip().str.upper()
                == str(customer_id).strip().upper()
            ]
            if matched.empty:
                return None
            return matched.to_dict(orient="records")[0]

        # ── Tool Definitions ──────────────────────────────────────────

        def tool_eda_narrative_summary() -> str:
            """Generates an executive-friendly narrative EDA summary. Use for any
            overview or 'summarize the data' request."""
            _log("Tool: tool_eda_narrative_summary")
            return generate_eda_summary(self.raw_df)

        def tool_eda_summary() -> str:
            """Full EDA report with raw numbers. Use only for specific statistical requests."""
            _log("Tool: tool_eda_summary")
            return str(run_eda(self.raw_df))

        def tool_dynamic_eda(metric_type: str, column_name: str = None) -> str:
            """Targeted EDA metric. metric_type: 'missing_values' | 'summary_stats' | 'correlation'."""
            _log(f"Tool: tool_dynamic_eda — {metric_type}, {column_name}")
            return str(query_eda_metrics(self.raw_df, metric_type, column_name))

        def tool_get_recommendation(segment_name: str, customer_id: str = None) -> str:
            """Cross-sell product recommendations for a segment, with optional per-customer tailoring."""
            _log(f"Tool: tool_get_recommendation — {segment_name}, {customer_id}")
            row = _find_customer_row(customer_id) if customer_id else None
            return str(get_cross_sell_recommendations(segment_name, row))

        def tool_get_profiles() -> str:
            """Segment-level averages and counts."""
            _log("Tool: tool_get_profiles")
            if self.segmented_df is not None:
                return str(get_segment_profiles(self.segmented_df))
            return "Data has not been segmented yet."

        def tool_get_personas() -> str:
            """Narrative customer personas with behavioral descriptions and marketing insights.
            Use when asked about 'personas', 'who are these customers', or 'customer profiles'."""
            _log("Tool: tool_get_personas")
            if self.segmented_df is None:
                return "Data has not been segmented yet."
            return str(generate_customer_personas(self.segmented_df))

        def tool_get_segment_insights() -> str:
            """Data-driven comparative insights about segments — findings, patterns,
            surprising discoveries. Use when asked about 'insights', 'findings',
            'what did you learn', or 'tell me about the segments'."""
            _log("Tool: tool_get_segment_insights")
            if self.segmented_df is None:
                return "Data has not been segmented yet."
            return generate_segment_insights(self.segmented_df)

        def tool_get_retention_strategies() -> str:
            """Segment-specific customer retention strategies. Use when asked about
            'retention', 'how to keep customers', 'prevent churn', or 'reduce attrition'."""
            _log("Tool: tool_get_retention_strategies")
            if self.segmented_df is None:
                return "Data has not been segmented yet."
            return get_retention_strategies(self.segmented_df)

        def tool_identify_at_risk_customers() -> str:
            """Finds customers showing churn or downgrade warning signs. Use when asked
            about 'at-risk customers', 'churn risk', 'who might leave', or 'retention risks'."""
            _log("Tool: tool_identify_at_risk_customers")
            if self.segmented_df is None:
                return "Data has not been segmented yet."
            return str(identify_at_risk_customers(self.segmented_df))

        def tool_get_model_evaluation() -> str:
            """Current segmentation evaluation metrics (silhouette, Davies-Bouldin, etc.)."""
            _log("Tool: tool_get_model_evaluation")
            return str(self.evaluation_metrics) if self.evaluation_metrics else "Not available yet."

        def tool_lookup_customer(customer_id: str) -> str:
            """Looks up a customer by ID. Returns NOT_FOUND if absent. Always call first."""
            _log(f"Tool: tool_lookup_customer — {customer_id}")
            row = _find_customer_row(customer_id)
            if row is None:
                return f"NOT_FOUND: Customer ID '{customer_id}' does NOT exist."
            return str(row)

        def tool_explain_customer(customer_id: str) -> str:
            """Explains WHY a customer is in their segment, with percentile context."""
            _log(f"Tool: tool_explain_customer — {customer_id}")
            row = _find_customer_row(customer_id)
            if row is None:
                return f"NOT_FOUND: Customer ID '{customer_id}' does NOT exist."
            if "Segment" not in row:
                return "Customer found, but data not segmented yet."
            return explain_customer_segment(row, self.segmented_df)

        def tool_run_segmentation(method: str = "rules", n_clusters: int = 3) -> str:
            """Runs/re-runs segmentation. method: 'rules', 'kmeans', 'hierarchical', 'dbscan'."""
            _log(f"Tool: tool_run_segmentation — {method}, {n_clusters}")
            if self.processed_df is None:
                _log("  -> Preprocessing first.")
                self.processed_df = preprocess_and_aggregate_customer_data(self.raw_df)
            self.segmented_df, self.evaluation_metrics = segment_customers(
                self.processed_df, method=method, n_clusters=n_clusters
            )
            self.last_filtered_df = self.segmented_df
            self.last_filter_description = "all customers (no filter)"
            counts = self.segmented_df["Segment"].value_counts().to_dict()
            return (
                f"Segmentation complete ({method}). Counts: {counts}. "
                f"Metrics: {self.evaluation_metrics}. CSV available in Segments tab."
            )

        def tool_compare_segmentation_methods(n_clusters: int = 3) -> str:
            """Compares KMeans/Hierarchical/DBSCAN metrics side by side."""
            _log(f"Tool: tool_compare_segmentation_methods — {n_clusters}")
            if self.processed_df is None:
                self.processed_df = preprocess_and_aggregate_customer_data(self.raw_df)
            return str(compare_clustering_methods(self.processed_df, n_clusters=n_clusters))

        def tool_find_upgrade_candidates() -> str:
            """Regular customers closest to Priority + conversion advice."""
            _log("Tool: tool_find_upgrade_candidates")
            if self.segmented_df is None:
                return "Data not segmented yet."
            candidates = find_upgrade_candidates(self.segmented_df, top_n=15)
            self.last_filtered_df = candidates
            self.last_filter_description = "Regular customers closest to Priority"
            if candidates.empty:
                return "No Regular-segment customers found."
            id_col = next(
                (c for c in candidates.columns if "cust" in c.lower() or "id" in c.lower()),
                candidates.columns[0],
            )
            bal_col = "avg_balance" if "avg_balance" in candidates.columns else "current_balance"
            cols = [c for c in [id_col, bal_col, "transaction_frequency", "proximity_score"] if c in candidates.columns]
            top_list = candidates[cols].round(2).to_dict(orient="records")
            advice = (
                "Encourage higher balances via targeted savings/fixed-deposit offers, "
                "and increase transaction engagement through cashback or bill-pay incentives."
            )
            return f"Top upgrade candidates: {top_list}\n\nRecommended action: {advice}"

        def tool_filter_last_segment(segment_tier: str) -> str:
            """Filters the most recent result by tier. Use for follow-ups like 'show only premium ones'."""
            _log(f"Tool: tool_filter_last_segment — {segment_tier}")
            base_df = self.last_filtered_df if self.last_filtered_df is not None else self.segmented_df
            if base_df is None or "Segment" not in base_df.columns:
                return "No prior segmentation result to filter."
            tier = str(segment_tier).strip().capitalize()
            filtered = base_df[base_df["Segment"] == tier]
            self.last_filtered_df = filtered
            self.last_filter_description = f"{tier} customers only"
            return f"Filtered to {len(filtered)} '{tier}' customers."

        def tool_get_business_kpis() -> str:
            """Headline business KPIs (customer counts, avg balance, cross-sell potential)."""
            _log("Tool: tool_get_business_kpis")
            if self.segmented_df is None:
                return "Data not segmented yet."
            return str(compute_business_kpis(self.segmented_df))

        def tool_identify_edge_cases() -> str:
            """Anomalous customers who don't fit cleanly into their segment."""
            _log("Tool: tool_identify_edge_cases")
            if self.segmented_df is None:
                return "Data not segmented yet."
            return str(identify_edge_cases(self.segmented_df))

        def tool_feature_importance() -> str:
            """Shows which features are most important for customer segmentation.
            Use when asked about 'feature importance', 'what drives segmentation',
            or 'most definitive features'."""
            _log("Tool: tool_feature_importance")
            if self.segmented_df is None:
                return "Data not segmented yet."
            return str(select_most_important_features(self.segmented_df))

        # ── Tool List ─────────────────────────────────────────────────
        self.tools = [
            tool_eda_narrative_summary,
            tool_eda_summary,
            tool_dynamic_eda,
            tool_get_recommendation,
            tool_get_profiles,
            tool_get_personas,
            tool_get_segment_insights,
            tool_get_retention_strategies,
            tool_identify_at_risk_customers,
            tool_get_model_evaluation,
            tool_lookup_customer,
            tool_explain_customer,
            tool_run_segmentation,
            tool_compare_segmentation_methods,
            tool_find_upgrade_candidates,
            tool_filter_last_segment,
            tool_get_business_kpis,
            tool_identify_edge_cases,
            tool_feature_importance,
        ]

        system_instruction = (
            "You are an AI-powered Analytics Agent for a Retail Bank.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. When asked about a specific customer ID, call 'tool_lookup_customer' FIRST. "
            "If NOT_FOUND, say so and stop. Never hallucinate.\n"
            "2. Only call tools needed for the specific query. Do NOT run full EDA or "
            "re-segmentation for narrow questions.\n"
            "3. For follow-ups referencing previous results ('show only premium ones'), "
            "use 'tool_filter_last_segment'.\n"
            "4. For 'why is customer X in segment Y', use 'tool_explain_customer'.\n"
            "5. For 'personas' or 'who are these customers', use 'tool_get_personas'.\n"
            "6. For 'insights', 'findings', 'what patterns do you see', use 'tool_get_segment_insights'.\n"
            "7. For 'retention', 'churn', 'keep customers', use 'tool_get_retention_strategies' "
            "and 'tool_identify_at_risk_customers'.\n"
            "8. For EDA 'summary' or 'overview', use 'tool_eda_narrative_summary'.\n"
            "9. For 'feature importance' or 'what drives segmentation', use 'tool_feature_importance'.\n"
            "10. For ambiguous queries ('best customers', 'which ones to target', 'important customers') "
            "without a stated metric, ask what metric they mean before answering."
        )

        self.chat = self.client.chats.create(
            model=MODEL_NAME if MODEL_NAME else "gemini-3.6-flash",
            config=types.GenerateContentConfig(
                tools=self.tools, system_instruction=system_instruction
            ),
        )

    def _check_ambiguous(self, query: str) -> str | None:
        """Human-in-the-loop: catches ambiguous queries before the LLM guesses."""
        q = query.lower()

        if _VAGUE_QUALIFIER.search(q) and not any(hint in q for hint in _METRIC_HINTS):
            return (
                "That's a bit ambiguous — do you mean customers with the highest "
                "**average balance**, highest **transaction frequency**, or highest "
                "**total spending**? Let me know which metric you'd like me to rank by."
            )

        if re.search(r"\bsegment\b", q) and "resegment" not in q and not any(k in q for k in _SEGMENT_KEYWORDS):
            if len(q.split()) <= 8:
                return (
                    "Sure — how would you like me to segment customers? For example: "
                    "rule-based thresholds on balance and transaction frequency, or "
                    "ML-based clustering (KMeans / Hierarchical / DBSCAN)?"
                )

        return None

    def run_default_pipeline(self, method: str = "rules") -> pd.DataFrame:
        """Runs the standard end-to-end pipeline on startup."""
        self.execution_log.append("Startup: preprocessing raw data...")
        self.processed_df = preprocess_and_aggregate_customer_data(self.raw_df)
        self.execution_log.append(f"Startup: running default segmentation ({method})...")
        self.segmented_df, self.evaluation_metrics = segment_customers(
            self.processed_df, method=method
        )
        self.last_filtered_df = self.segmented_df
        self.last_filter_description = "all customers (no filter)"
        self.execution_log.append("Startup: pipeline complete.")
        return self.segmented_df

    def process_query(self, user_query: str) -> dict:
        """Processes a natural language query. Returns answer + execution log."""
        self.execution_log = []
        self.execution_log.append(f'Received query: "{user_query}"')

        clarification = self._check_ambiguous(user_query)
        if clarification:
            self.execution_log.append("Query flagged as ambiguous — requesting clarification.")
            return {"answer": clarification, "log": list(self.execution_log)}

        try:
            self.execution_log.append("Dispatching to Gemini agent...")
            response = self.chat.send_message(user_query)
            self.execution_log.append("Agent finished.")
            return {"answer": response.text, "log": list(self.execution_log)}
        except Exception as e:
            self.execution_log.append(f"Error: {str(e)}")
            return {"answer": f"Error: {str(e)}", "log": list(self.execution_log)}