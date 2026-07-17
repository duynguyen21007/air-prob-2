import sys
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# Add the root directory to sys.path so we can import from src
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from src.config import SAMPLE_IDS, INPUT_DIR, DATA_DIR, MOCK_LLM
from src.retrieval import Icd10HybridSearcher

STAGE4_DIR = DATA_DIR / "stage4_rxnorm"
STAGE5_DIR = DATA_DIR / "stage5_icd10"
STAGE5_DIR.mkdir(parents=True, exist_ok=True)


class CompactPositionEncoder(json.JSONEncoder):
    """JSON encoder that keeps short lists (like position arrays) on one line."""
    def encode(self, o):
        return self._encode(o, indent_level=0)

    def _encode(self, o, indent_level):
        indent = "  "
        if isinstance(o, dict):
            if not o:
                return "{}"
            items = []
            for k, v in o.items():
                encoded_value = self._encode(v, indent_level + 1)
                items.append(f'{indent * (indent_level + 1)}"{k}": {encoded_value}')
            return "{\n" + ",\n".join(items) + "\n" + indent * indent_level + "}"
        elif isinstance(o, list):
            # Keep short lists of primitives (like position) on one line
            if all(isinstance(item, (int, float)) for item in o):
                return "[" + ", ".join(json.dumps(item) for item in o) + "]"
            if not o:
                return "[]"
            items = []
            for item in o:
                encoded_value = self._encode(item, indent_level + 1)
                items.append(f'{indent * (indent_level + 1)}{encoded_value}')
            return "[\n" + ",\n".join(items) + "\n" + indent * indent_level + "]"
        else:
            return json.dumps(o, ensure_ascii=False)


def run_stage5(is_mock=False):
    if not STAGE4_DIR.exists():
        print("Stage 4 output not found. Please run stage 4 first.")
        return

    use_mock = is_mock or MOCK_LLM or os.getenv("MOCK_LLM", "false").lower() in ("true", "1", "yes")

    data_csv_path = BASE_DIR / "data_icds.csv"
    chroma_persist_dir = DATA_DIR / "chroma_icd10_db"
    
    if not chroma_persist_dir.exists():
        print(f"ChromaDB not found at {chroma_persist_dir}! Please run `python scripts/build_icd10_index.py` first.")
        return
        
    if use_mock:
        print("[MOCK MODE] Stage 5 ICD-10 using local BGE CrossEncoder reranking (without LLM)")
    else:
        print("[LLM MODE] Stage 5 ICD-10 using vLLM candidate reranking")

    searcher = Icd10HybridSearcher(str(data_csv_path), str(chroma_persist_dir))

    json_files = list(STAGE4_DIR.glob("*.json"))
    lookup = {}
    for in_file in tqdm(json_files, desc="Processing Stage 5 ICD-10"):
        doc_id = in_file.stem
        out_file = STAGE5_DIR / f"{doc_id}.json"
        
        if not in_file.exists():
            continue
            
        if out_file.exists():
            print(f"Skipping {doc_id} as it is already processed.")
            continue
            
        with open(in_file, "r", encoding="utf-8") as f:
            stage4_data = json.load(f)
            
        # 1. Collect unique CHẨN_ĐOÁN strings
        diagnoses = set()
        for ent in stage4_data:
            if ent["type"] == "CHẨN_ĐOÁN":
                diagnoses.add(ent["text"])
                
        # 2. Get ICD-10 candidates using hybrid search + LLM reranking
        if diagnoses:
            for diag in diagnoses:
                if diag in lookup:
                    continue
                try:
                    if use_mock:
                        # Non-LLM local BGE CrossEncoder reranking
                        qualified_icds = searcher.get_qualified_icds(diag, margin=0.05, absolute_threshold=0.0)
                    else:
                        # vLLM reranker
                        qualified_icds = searcher.get_qualified_icds_v2(diag)
                except Exception as e:
                    print(f"Warning: Reranking failed for '{diag}' ({e}). Falling back to local search.")
                    qualified_icds = searcher.get_qualified_icds(diag, margin=0.05, absolute_threshold=0.0)

                if qualified_icds:
                    lookup[diag] = qualified_icds
                
        # 3. Now rebuild entities with candidates
        entities = []
        for ent in stage4_data:
            new_ent = dict(ent) # Make a copy
            if new_ent["type"] == "CHẨN_ĐOÁN":
                icd_codes = lookup.get(new_ent["text"])
                if icd_codes:
                    new_ent["candidates"] = icd_codes
                else:
                    new_ent["candidates"] = []
            
            entities.append(new_ent)
            
        # Save as bare array (contest format)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(CompactPositionEncoder().encode(entities) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 5 ICD-10 Retrieval")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (using local BGE reranker without LLM)")
    args = parser.parse_args()

    run_stage5(is_mock=args.mock)
