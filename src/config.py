import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DATASET_PATH = os.getenv("DATASET_PATH", "data/bank_transactions.csv")

# Current stable Gemini flash model
MODEL_NAME = "gemini-3.6-flash"

# Optional: cap dataset size for fast live demos on the full 1M+ row Kaggle
# dataset. Set DEFAULT_SAMPLE_SIZE=0 (or leave unset) to use the full file.
DEFAULT_SAMPLE_SIZE = int(os.getenv("DEFAULT_SAMPLE_SIZE", "100000"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")