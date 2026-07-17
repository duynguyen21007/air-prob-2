import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"
INPUT_DIR = BASE_DIR / "input"
DATA_DIR = BASE_DIR / "data"
MOCK_DATA_DIR = BASE_DIR / "mock_data"

# Load config
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

SAMPLE_IDS = config.get("sample_ids", [])

# vLLM configuration (via OpenAI-compatible API)
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8211/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "dummy")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-9B")
VLLM_MAX_TOKENS = int(os.getenv("VLLM_MAX_TOKENS", "4096"))
VLLM_TEMPERATURE = float(os.getenv("VLLM_TEMPERATURE", "0.0"))

# Mock mode flag (run pipeline without LLM server)
MOCK_LLM = os.getenv("MOCK_LLM", "false").lower() in ("true", "1", "yes")
