import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Paths
BASE_DIR = Path(__file__).parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"
INPUT_DIR = BASE_DIR / "input"
DATA_DIR = BASE_DIR / "data"

# Load config
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

SAMPLE_IDS = config.get("sample_ids", [])
MODEL_NAME = config.get("model", "gemini-2.5-flash")
TEMPERATURE = config.get("temperature", 0.1)
MAX_RETRIES = config.get("max_retries", 3)
RETRY_BASE_DELAY = config.get("retry_base_delay", 2.0)
RPM_LIMIT = config.get("rpm_limit", 15)
