import pandas as pd
from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, MODEL_NAME
from src.tools.eda_tool import run_eda
from src.tools.feature_engineering import preprocess_and_aggregate_customer_data
from src.tools.segmentation_tool import segment_customers
from src.tools.explainability_tool import get_segment_profiles, get_cross_sell_recommendations

class RetailBankingAgent:
    def __init__(self, raw_df: pd.DataFrame):
        self.raw_df = raw_df
        self.processed_df = None
        self.segmented_df = None
        self.evaluation_metrics = {}
        
        # Initialize Google GenAI Client
        self.client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else genai.Client()
        
        # 1. High-level EDA summary tool
        def tool_eda_summary() -> str:
            """Generates a summary of the exploratory data analysis including missing values and key statistics."""
            return str(run_eda(self.raw_df))

        # 2. Segment recommendation tool
        def tool_get_recommendation(segment_name: str) -> str:
            """Provides financial product recommendations given a segment name such as Priority, Regular, or Dormant."""
            return str(get_cross_sell_recommendations(segment_name))

        # 3. Segment profile tool
        def tool_get_profiles() -> str:
            """Retrieves demographic and financial profile metrics for each customer segment."""
            if self.segmented_df is not None:
                return str(get_segment_profiles(self.segmented_df))
            return "Data has not been segmented yet."

        # 4. Model evaluation tool
        def tool_get_model_evaluation() -> str:
            """Retrieves clustering model evaluation metrics such as Silhouette Score and Inertia."""
            if self.evaluation_metrics:
                return str(self.evaluation_metrics)
            return "Model evaluation metrics are not available yet."

        # 5. Specific Customer Lookup Tool (Fixes Non-Existent User Bug)
        def tool_lookup_customer(customer_id: str) -> str:
            """Looks up a specific customer by Customer ID in the dataset to check if they exist and return their segment and details."""
            df_to_search = self.segmented_df if self.segmented_df is not None else self.raw_df
            
            # Identify customer ID column dynamically
            id_col = next((col for col in df_to_search.columns if 'cust' in col.lower() or 'id' in col.lower()), df_to_search.columns[0])
            
            # Search for exact match
            matched_row = df_to_search[df_to_search[id_col].astype(str).str.strip().str.upper() == str(customer_id).strip().upper()]
            
            if matched_row.empty:
                return f"NOT_FOUND: Customer ID '{customer_id}' does NOT exist in the database."
            
            return str(matched_row.to_dict(orient="records")[0])

        self.tools = [
            tool_eda_summary,
            tool_get_recommendation,
            tool_get_profiles,
            tool_get_model_evaluation,
            tool_lookup_customer
        ]
        
        system_instruction = (
            "You are an AI-powered Analytics Agent for a Retail Bank.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. When asked about a specific customer ID or user, you MUST call 'tool_lookup_customer' first. "
            "If the tool returns 'NOT_FOUND', state clearly that the customer does not exist in the database and stop. "
            "Do NOT make up or hallucinate details for non-existent customers.\n"
            "2. HUMAN-IN-THE-LOOP RULE: If a user query is vague or ambiguous (e.g., asking to segment without specifying parameters), "
            "ask relevant clarifying questions before proceeding."
        )
        
        # Start chat session with native function-calling configuration
        self.chat = self.client.chats.create(
            model=MODEL_NAME if MODEL_NAME else "gemini-2.5-flash",
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