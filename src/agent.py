import pandas as pd
from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, MODEL_NAME
from src.tools.eda_tool import run_eda, query_eda_metrics
from src.tools.feature_engineering import preprocess_and_aggregate_customer_data
from src.tools.segmentation_tool import segment_customers, find_upgrade_candidates
from src.tools.explainability_tool import get_segment_profiles, get_cross_sell_recommendations, explain_customer_segment


class RetailBankingAgent:
    def __init__(self, raw_df: pd.DataFrame):
        self.raw_df = raw_df
        self.processed_df = None
        self.segmented_df = None
        self.evaluation_metrics = {}

        # Initialize Google GenAI Client
        self.client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else genai.Client()

        def _find_customer_row(customer_id: str):
            """Internal helper: locates a customer row as a dict, or None if not found."""
            df_to_search = self.segmented_df if self.segmented_df is not None else self.raw_df
            id_col = next((col for col in df_to_search.columns if 'cust' in col.lower() or 'id' in col.lower()), df_to_search.columns[0])
            matched_row = df_to_search[df_to_search[id_col].astype(str).str.strip().str.upper() == str(customer_id).strip().upper()]
            if matched_row.empty:
                return None
            return matched_row.to_dict(orient="records")[0]

        # 1. High-level EDA summary tool
        def tool_eda_summary() -> str:
            """Generates a full summary of the exploratory data analysis including missing values and key statistics. Use for broad/general EDA requests."""
            return str(run_eda(self.raw_df))

        # 1b. Targeted EDA tool (dynamic EDA from a specific user query)
        def tool_dynamic_eda(metric_type: str, column_name: str = None) -> str:
            """
            Runs ONE targeted EDA metric instead of the full report. Use this when the
            user asks a specific EDA question rather than "give me a full analysis".
            metric_type must be one of: 'missing_values', 'summary_stats', 'correlation'.
            column_name is required only for 'summary_stats'.
            """
            return str(query_eda_metrics(self.raw_df, metric_type, column_name))

        # 2. Segment recommendation tool
        def tool_get_recommendation(segment_name: str) -> str:
            """Provides financial product recommendations given a segment name such as Priority, Regular, or Dormant."""
            return str(get_cross_sell_recommendations(segment_name))

        # 3. Segment profile tool
        def tool_get_profiles() -> str:
            """Retrieves demographic and financial profile metrics (segment-level averages) for each customer segment."""
            if self.segmented_df is not None:
                return str(get_segment_profiles(self.segmented_df))
            return "Data has not been segmented yet."

        # 4. Model evaluation tool
        def tool_get_model_evaluation() -> str:
            """Retrieves clustering/segmentation evaluation metrics such as Silhouette Score, Inertia, or rule thresholds used."""
            if self.evaluation_metrics:
                return str(self.evaluation_metrics)
            return "Model evaluation metrics are not available yet."

        # 5. Specific Customer Lookup Tool
        def tool_lookup_customer(customer_id: str) -> str:
            """Looks up a specific customer by Customer ID in the dataset to check if they exist and return their segment and details."""
            row = _find_customer_row(customer_id)
            if row is None:
                return f"NOT_FOUND: Customer ID '{customer_id}' does NOT exist in the database."
            return str(row)

        # 6. Per-customer explainability tool
        def tool_explain_customer(customer_id: str) -> str:
            """
            Explains WHY a specific customer was placed into their current segment,
            tied to their actual balance/frequency values. Use for queries like
            "Why is customer X flagged as Priority?" or "Is customer ID 4521 Dormant, and why?".
            """
            row = _find_customer_row(customer_id)
            if row is None:
                return f"NOT_FOUND: Customer ID '{customer_id}' does NOT exist in the database."
            if "Segment" not in row:
                return "Customer found, but data has not been segmented yet."
            return explain_customer_segment(row)

        # 7. Dynamic (re-)segmentation tool
        def tool_run_segmentation(method: str = "rules", n_clusters: int = 3) -> str:
            """
            Runs (or re-runs) customer segmentation on demand. Use this whenever the user
            explicitly asks to segment/re-segment customers, or asks to change the
            segmentation approach (e.g. "segment customers using KMeans with 4 clusters",
            or "segment customers based on balance and transaction frequency").
            method must be 'rules' or 'kmeans'.
            """
            if self.processed_df is None:
                self.processed_df = preprocess_and_aggregate_customer_data(self.raw_df)
            self.segmented_df, self.evaluation_metrics = segment_customers(
                self.processed_df, method=method, n_clusters=n_clusters
            )
            counts = self.segmented_df["Segment"].value_counts().to_dict()
            return (
                f"Segmentation complete using method='{method}'. "
                f"Segment counts: {counts}. Evaluation metrics: {self.evaluation_metrics}. "
                f"A downloadable CSV of all customers and their segments is available in the "
                f"'Customer Segments' tab of the app."
            )

        # 8. Regular -> Priority upgrade-candidate tool
        def tool_find_upgrade_candidates() -> str:
            """
            Finds Regular-segment customers closest to crossing into the Priority tier
            and recommends how to convert them. Use for queries like "Which regular
            customers can be converted to priority customers? What should be done for the same?"
            """
            if self.segmented_df is None:
                return "Data has not been segmented yet."
            candidates = find_upgrade_candidates(self.segmented_df, top_n=15)
            if candidates.empty:
                return "No Regular-segment customers found (or dataset not segmented)."

            id_col = next((c for c in candidates.columns if 'cust' in c.lower() or 'id' in c.lower()), candidates.columns[0])
            bal_col = "avg_balance" if "avg_balance" in candidates.columns else "current_balance"
            cols_to_show = [c for c in [id_col, bal_col, "transaction_frequency", "proximity_score"] if c in candidates.columns]
            top_list = candidates[cols_to_show].round(2).to_dict(orient="records")

            advice = (
                "To convert these customers: encourage higher average balances via targeted "
                "savings/fixed-deposit offers, and increase transaction engagement through "
                "cashback or bill-pay incentives to help them cross the Priority threshold "
                "(balance > ₹50,000, or balance > ₹20,000 with 10+ transactions)."
            )
            return f"Top upgrade candidates: {top_list}\n\nRecommended action: {advice}"

        self.tools = [
            tool_eda_summary,
            tool_dynamic_eda,
            tool_get_recommendation,
            tool_get_profiles,
            tool_get_model_evaluation,
            tool_lookup_customer,
            tool_explain_customer,
            tool_run_segmentation,
            tool_find_upgrade_candidates,
        ]

        system_instruction = (
            "You are an AI-powered Analytics Agent for a Retail Bank.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. When asked about a specific customer ID or user, you MUST call 'tool_lookup_customer' first. "
            "If the tool returns 'NOT_FOUND', state clearly that the customer does not exist in the database and stop. "
            "Do NOT make up or hallucinate details for non-existent customers.\n"
            "2. Only call the tools that are actually needed for the specific query — do not run a full EDA or "
            "re-segmentation if the user only asked a narrow question (e.g. a single customer lookup or a "
            "targeted stat). Prefer 'tool_dynamic_eda' over 'tool_eda_summary' for narrow EDA questions.\n"
            "3. HUMAN-IN-THE-LOOP RULE: If a user query is vague or ambiguous (e.g., asking to segment without "
            "specifying parameters or thresholds), ask relevant clarifying questions before proceeding.\n"
            "4. When asked why a customer belongs to a segment, use 'tool_explain_customer', not 'tool_get_profiles'."
        )

        # Start chat session with native function-calling configuration
        self.chat = self.client.chats.create(
            model=MODEL_NAME if MODEL_NAME else "gemini-3.6-flash",
            config=types.GenerateContentConfig(
                tools=self.tools,
                system_instruction=system_instruction
            )
        )

    def run_default_pipeline(self, method: str = "rules") -> pd.DataFrame:
        """Runs the standard end-to-end data transformation, segmentation, and evaluation pipeline on startup."""
        self.processed_df = preprocess_and_aggregate_customer_data(self.raw_df)
        self.segmented_df, self.evaluation_metrics = segment_customers(self.processed_df, method=method)
        return self.segmented_df

    def process_query(self, user_query: str) -> str:
        """Processes natural language user requests using Gemini Agent."""
        try:
            response = self.chat.send_message(user_query)
            return response.text
        except Exception as e:
            return f"An error occurred while processing the request: {str(e)}"