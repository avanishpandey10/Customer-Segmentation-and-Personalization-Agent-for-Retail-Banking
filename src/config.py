import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DATASET_PATH = os.getenv("DATASET_PATH", "data/bank_transactions.csv")

# Updated to the current standard stable flash model
MODEL_NAME = "gemini-3.6-flash"