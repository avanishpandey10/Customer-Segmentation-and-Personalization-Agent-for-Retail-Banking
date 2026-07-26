import re
import pandas as pd
from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, MODEL_NAME
from src.logger_setup import get_logger
from src.tools.eda_tool import run_eda, query_eda_metrics
from src.tools.feature_engineering import preprocess_and_aggregate_customer_data
from src.tools.segmentation_tool import segment_customers, find_upgrade_candidates, compare_clustering_methods
from src.tools.explainability_tool import get_segment_profiles, get_cross_sell_recommendations, explain_customer_segment
from src.tools.kpi_tool import compute_business_kpis

logger = get_logger(__name__)

# Ambiguous phrasing that MUST trigger a clarifying question rather than a
# guessed answer (guaranteed in code, not left to the LLM's discretion —
# this is the hackathon's "human-in-the-loop" requirement).
_VAGUE_QUALIFIER = re.compile(r"\b(best|top|good|great|ideal)\b", re.IGNORECASE)
_METRIC_HINTS = [
    "balance", "frequency", "spend", "transaction", "priority", "regular",
    "dormant", "amount", "count", "cluster", "kmeans", "recent", "frequency",
    "recency", "trend",
]
_SEGMENT_KEYWORDS = ["balance", "frequency", "kmeans", "cluster", "rule", "dbscan", "hierarchical", "spend"]


class RetailBankingAgent:
    def __init__(self, raw_df: pd.DataFrame):
        self.raw_df = raw_df
        self.processed_df = None
        self.segmented_df = None
        self.evaluation_metrics = {}
        self.execution_log = []

        # Simple conversation memory: remembers the last filtered view so
        # follow-ups like "show only premium ones" don't need to re-run
        # the whole pipeline from scratch.
        self.last_filtered_df = None
        self.last_filter_description = None

        self.client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else genai.Client()

        def _log(step: str):
            self.execution_log.append(step)
            logger.info(step)

        def _find_customer_row(customer_id: str):
            df_to_search = self.segmented_df if self.segmented_df is not None else self.raw_df
            id_col = next((c for c in df_to_search.columns if 'cust' in c.lower() or 'id' in c.lower()), df_to_search.columns[0])
            matched = df_to_search[df_to_search[id_col].astype(str).str.strip().str.upper() == str(customer_id).strip().upper()]
            if matched.empty:
                return None
            return matched.to_dict(orient="records")[0]

        # 1. Full EDA
        def tool_eda_summary() -> str:
            """Generates a full EDA report (missing values, dtypes, numeric summary). Use for broad/general EDA requests."""
            _log("Tool: tool_eda_summary — running full EDA report.")
            return str(run_eda(self.raw_df))

        # 2. Targeted EDA
        def tool_dynamic_eda(metric_type: str, column_name: str = None) -> str:
            """
            Runs ONE targeted EDA metric instead of the full report — use for a specific
            EDA question. metric_type: 'missing_values' | 'summary_stats' | 'correlation'.
            column_name is required only for 'summary_stats'.
            """
            _log(f"Tool: tool_dynamic_eda — metric_type={metric_type}, column_name={column_name}.")
            return str(query_eda_metrics(self.raw_df, metric_type, column_name))

        # 3. Cross-sell recommendation
        def tool_get_recommendation(segment_name: str, customer_id: str = None) -> str:
            """
            Provides financial product recommendations for a segment (Priority/Regular/Dormant).
            If customer_id is given, adds secondary rules specific to that customer.
            """
            _log(f"Tool: tool_get_recommendation — segment={segment_name}, customer_id={customer_id}.")
            row = _find_customer_row(customer_id) if customer_id else None
            return str(get_cross_sell_recommendations(segment_name, row))

        # 4. Segment profiles
        def tool_get_profiles() -> str:
            """Retrieves segment-level averages and counts for each customer segment."""
            _log("Tool: tool_get_profiles — computing segment-level averages.")
            if self.segmented_df is not None:
                return str(get_segment_profiles(self.segmented_df))
            return "Data has not been segmented yet."

        # 5. Model evaluation
        def tool_get_model_evaluation() -> str:
            """Retrieves the current segmentation's evaluation metrics (silhouette, Davies-Bouldin, Calinski-Harabasz, or rule thresholds)."""
            _log("Tool: tool_get_model_evaluation.")
            return str(self.evaluation_metrics) if self.evaluation_metrics else "Model evaluation metrics are not available yet."

        # 6. Customer lookup
        def tool_lookup_customer(customer_id: str) -> str:
            """Looks up a specific customer by Customer ID; returns NOT_FOUND if they don't exist. Always call this before discussing a specific customer."""
            _log(f"Tool: tool_lookup_customer — customer_id={customer_id}.")
            row = _find_customer_row(customer_id)
            if row is None:
                return f"NOT_FOUND: Customer ID '{customer_id}' does NOT exist in the database."
            return str(row)

        # 7. Per-customer explainability
        def tool_explain_customer(customer_id: str) -> str:
            """Explains WHY a specific customer was placed into their current segment, with a percentile/ratio comparison against Regular-segment customers."""
            _log(f"Tool: tool_explain_customer — customer_id={customer_id}.")
            row = _find_customer_row(customer_id)
            if row is None:
                return f"NOT_FOUND: Customer ID '{customer_id}' does NOT exist in the database."
            if "Segment" not in row:
                return "Customer found, but data has not been segmented yet."
            return explain_customer_segment(row, self.segmented_df)

        # 8. Dynamic (re-)segmentation
        def tool_run_segmentation(method: str = "rules", n_clusters: int = 3) -> str:
            """
            Runs (or re-runs) customer segmentation on demand. method: 'rules', 'kmeans',
            'hierarchical', or 'dbscan'. Use whenever the user asks to segment/re-segment
            customers or change the segmentation approach.
            """
            _log(f"Tool: tool_run_segmentation — method={method}, n_clusters={n_clusters}.")
            if self.processed_df is None:
                _log("  -> No processed data yet; running preprocessing first.")
                self.processed_df = preprocess_and_aggregate_customer_data(self.raw_df)
            self.segmented_df, self.evaluation_metrics = segment_customers(self.processed_df, method=method, n_clusters=n_clusters)
            self.last_filtered_df = self.segmented_df
            self.last_filter_description = "all customers (no filter)"
            counts = self.segmented_df["Segment"].value_counts().to_dict()
            _log(f"  -> Segmentation complete. Counts: {counts}")
            return (
                f"Segmentation complete using method='{method}'. Segment counts: {counts}. "
                f"Evaluation metrics: {self.evaluation_metrics}. A downloadable CSV is available "
                f"in the 'Customer Segments' tab of the app."
            )

        # 9. Compare clustering methods
        def tool_compare_segmentation_methods(n_clusters: int = 3) -> str:
            """
            Runs KMeans, Hierarchical, and DBSCAN and compares their evaluation metrics
            (silhouette, Davies-Bouldin, Calinski-Harabasz) side by side. Use when the
            user asks which clustering algorithm works best, or wants a comparison.
            """
            _log(f"Tool: tool_compare_segmentation_methods — n_clusters={n_clusters}.")
            if self.processed_df is None:
                self.processed_df = preprocess_and_aggregate_customer_data(self.raw_df)
            return str(compare_clustering_methods(self.processed_df, n_clusters=n_clusters))

        # 10. Upgrade candidates
        def tool_find_upgrade_candidates() -> str:
            """Finds Regular-segment customers closest to crossing into the Priority tier and recommends how to convert them."""
            _log("Tool: tool_find_upgrade_candidates.")
            if self.segmented_df is None:
                return "Data has not been segmented yet."
            candidates = find_upgrade_candidates(self.segmented_df, top_n=15)
            self.last_filtered_df = candidates
            self.last_filter_description = "Regular customers closest to Priority (upgrade candidates)"
            if candidates.empty:
                return "No Regular-segment customers found (or dataset not segmented)."
            id_col = next((c for c in candidates.columns if 'cust' in c.lower() or 'id' in c.lower()), candidates.columns[0])
            bal_col = "avg_balance" if "avg_balance" in candidates.columns else "current_balance"
            cols = [c for c in [id_col, bal_col, "transaction_frequency", "proximity_score"] if c in candidates.columns]
            top_list = candidates[cols].round(2).to_dict(orient="records")
            advice = (
                "Encourage higher average balances via targeted savings/fixed-deposit offers, "
                "and increase transaction engagement through cashback or bill-pay incentives to help "
                "them cross the Priority threshold (balance > ₹50,000, or balance > ₹20,000 with 10+ transactions)."
            )
            return f"Top upgrade candidates: {top_list}\n\nRecommended action: {advice}"

        # 11. Filter last result — lightweight conversation memory
        def tool_filter_last_segment(segment_tier: str) -> str:
            """
            Filters the MOST RECENT segmentation result down to one tier, without
            re-running the whole pipeline. Use for follow-ups like "show only the
            premium/priority ones" that refer back to a segmentation already done
            earlier in this conversation. segment_tier: 'Priority', 'Regular', or 'Dormant'.
            """
            _log(f"Tool: tool_filter_last_segment — segment_tier={segment_tier} (using conversation memory).")
            base_df = self.last_filtered_df if self.last_filtered_df is not None else self.segmented_df
            if base_df is None or "Segment" not in base_df.columns:
                return "No prior segmentation result to filter — please segment customers first."
            tier = str(segment_tier).strip().capitalize()
            filtered = base_df[base_df["Segment"] == tier]
            self.last_filtered_df = filtered
            self.last_filter_description = f"{tier} customers only"
            return f"Filtered to {len(filtered)} '{tier}' customers from the previous result. Sample: {filtered.head(5).to_dict(orient='records')}"

        # 12. Business KPIs
        def tool_get_business_kpis() -> str:
            """Returns headline business KPIs (total customers, segment counts, avg balance, avg spend, cross-sell potential) for a summary-style query."""
            _log("Tool: tool_get_business_kpis.")
            if self.segmented_df is None:
                return "Data has not been segmented yet."
            return str(compute_business_kpis(self.segmented_df))

        self.tools = [
            tool_eda_summary,
            tool_dynamic_eda,
            tool_get_recommendation,
            tool_get_profiles,
            tool_get_model_evaluation,
            tool_lookup_customer,
            tool_explain_customer,
            tool_run_segmentation,
            tool_compare_segmentation_methods,
            tool_find_upgrade_candidates,
            tool_filter_last_segment,
            tool_get_business_kpis,
        ]

        system_instruction = (
            "You are an AI-powered Analytics Agent for a Retail Bank.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. When asked about a specific customer ID, you MUST call 'tool_lookup_customer' first. "
            "If it returns 'NOT_FOUND', say so clearly and stop. Never hallucinate customer details.\n"
            "2. Only call the tools actually needed for the specific query — do not run a full EDA or "
            "re-segmentation if the user only asked a narrow question.\n"
            "3. If the user refers to a previous result ('show only the premium ones', 'filter those down'), "
            "use 'tool_filter_last_segment' instead of re-running segmentation from scratch.\n"
            "4. When asked why a customer belongs to a segment, use 'tool_explain_customer'.\n"
            "5. For 'best'/'top'/'good' customers without a stated metric, ask which metric "
            "(balance, frequency, or spend) they mean before answering — do not guess."
        )

        self.chat = self.client.chats.create(
            model=MODEL_NAME if MODEL_NAME else "gemini-3.6-flash",
            config=types.GenerateContentConfig(tools=self.tools, system_instruction=system_instruction)
        )

    def _check_ambiguous(self, query: str) -> str | None:
        """
        Guarantees human-in-the-loop clarification for ambiguous queries in code,
        rather than relying solely on the LLM following a prompt instruction.
        """
        q = query.lower()

        if _VAGUE_QUALIFIER.search(q) and not any(hint in q for hint in _METRIC_HINTS):
            return (
                "That's a bit ambiguous — do you mean customers with the highest **average balance**, "
                "highest **transaction frequency**, or highest **total spending**? "
                "Let me know which metric you'd like me to rank by."
            )

        if re.search(r"\bsegment\b", q) and "resegment" not in q and not any(k in q for k in _SEGMENT_KEYWORDS):
            # Only trigger for genuinely bare requests like "segment the customers"
            if len(q.split()) <= 6:
                return (
                    "Sure — how would you like me to segment customers? For example: rule-based "
                    "thresholds on balance and transaction frequency, or ML-based clustering "
                    "(KMeans / Hierarchical / DBSCAN)?"
                )

        return None

    def run_default_pipeline(self, method: str = "rules") -> pd.DataFrame:
        """Runs the standard end-to-end data transformation, segmentation, and evaluation pipeline on startup."""
        self.execution_log.append("Startup: preprocessing raw data.")
        self.processed_df = preprocess_and_aggregate_customer_data(self.raw_df)
        self.execution_log.append(f"Startup: running default segmentation (method={method}).")
        self.segmented_df, self.evaluation_metrics = segment_customers(self.processed_df, method=method)
        self.last_filtered_df = self.segmented_df
        self.last_filter_description = "all customers (no filter)"
        self.execution_log.append("Startup: pipeline complete.")
        return self.segmented_df

    def process_query(self, user_query: str) -> dict:
        """
        Processes a natural language user request. Returns a dict with the
        final 'answer' text and an 'log' list of the steps taken, so the UI
        can show visible agent reasoning/progress.
        """
        self.execution_log = []
        self.execution_log.append(f"Received query: \"{user_query}\"")

        clarification = self._check_ambiguous(user_query)
        if clarification:
            self.execution_log.append("Query flagged as ambiguous — requesting clarification (human-in-the-loop) instead of guessing.")
            return {"answer": clarification, "log": list(self.execution_log)}

        try:
            self.execution_log.append("Dispatching to Gemini agent for intent parsing and dynamic tool planning...")
            response = self.chat.send_message(user_query)
            self.execution_log.append("Agent finished tool orchestration and generated the final response.")
            return {"answer": response.text, "log": list(self.execution_log)}
        except Exception as e:
            self.execution_log.append(f"Error: {str(e)}")
            return {"answer": f"An error occurred while processing the request: {str(e)}", "log": list(self.execution_log)}