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
from src.retrieval.rxnorm_hybrid_search import RxNormHybridSearcher

STAGE1_DIR = DATA_DIR / "stage1_ner"
STAGE4_OUT_DIR = DATA_DIR / "stage4_rxnorm"


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
            # Keep short lists of strings (like assertions or candidates) on one line
            if all(isinstance(item, str) for item in o) and len(o) <= 3:
                return "[" + ", ".join(json.dumps(item, ensure_ascii=False) for item in o) + "]"
            if not o:
                return "[]"
            items = []
            for item in o:
                items.append(indent * (indent_level + 1) + self._encode(item, indent_level + 1))
            return "[\n" + ",\n".join(items) + "\n" + indent * indent_level + "]"
        else:
            return json.dumps(o, ensure_ascii=False)


def run_stage4(is_mock=False):
    STAGE4_OUT_DIR.mkdir(parents=True, exist_ok=True)
    use_mock = is_mock or MOCK_LLM or os.getenv("MOCK_LLM", "false").lower() in ("true", "1", "yes")
    
    data_csv_path = BASE_DIR / "data_rxnorm.csv"
    chroma_persist_dir = DATA_DIR / "chroma_rxnorm_db"
    
    if not chroma_persist_dir.exists():
        print(f"ChromaDB not found at {chroma_persist_dir}! Please run `python scripts/build_rxnorm_index.py` first.")
        return

    if use_mock:
        print("[MOCK MODE] Stage 4 RxNorm using local BGE CrossEncoder reranking (without LLM)")
    else:
        print("[LLM MODE] Stage 4 RxNorm using vLLM candidate reranking")

    searcher = RxNormHybridSearcher(str(data_csv_path), str(chroma_persist_dir))
    
    json_files = list(STAGE1_DIR.glob("*.json"))
    lookup = {}
    for stage1_file in tqdm(json_files, desc="Processing Stage 4 RxNorm"):
        doc_id = stage1_file.stem
        out_file = STAGE4_OUT_DIR / f"{doc_id}.json"
        
        if not stage1_file.exists():
            print(f"Warning: Stage 1 output {stage1_file} does not exist. Skipping.")
            continue
            
        if out_file.exists():
            print(f"Skipping {doc_id} as it is already processed.")
            continue
            
        with open(stage1_file, "r", encoding="utf-8") as f:
            stage1_data = json.load(f)
            
        # Extract THUỐC entities
        thuoc_entities = [ent for ent in stage1_data if ent["type"] == "THUỐC"]
        
        if thuoc_entities:
            raw_drugs = [ent["text"] for ent in thuoc_entities]
            raw_drugs = list(set(raw_drugs))
            
            for drug in raw_drugs:
                if drug in lookup:
                    continue
                try:
                    if use_mock:
                        # Non-LLM local BGE CrossEncoder reranking
                        rxcuis = searcher.get_qualified_rxcuis(drug, margin=0.05, absolute_threshold=0.0)
                    else:
                        # vLLM reranker
                        rxcuis = searcher.get_qualified_rxcuis_v2(drug)
                except Exception as e:
                    print(f"Warning: Reranking failed for '{drug}' ({e}). Falling back to local search.")
                    rxcuis = searcher.get_qualified_rxcuis(drug, margin=0.05, absolute_threshold=0.0)
                    
                if rxcuis:
                    lookup[drug] = rxcuis[:5]
                
        # Now rebuild entities with candidates
        entities = []
        for ent in stage1_data:
            new_ent = dict(ent) # Make a copy
            if new_ent["type"] == "THUỐC":
                rxcuis = lookup.get(new_ent["text"])
                if rxcuis:
                    new_ent["candidates"] = rxcuis
                else:
                    new_ent["candidates"] = []
            
            entities.append(new_ent)
            
        # Save as bare array (contest format)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(CompactPositionEncoder().encode(entities) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 4 RxNorm Retrieval")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (using local BGE reranker without LLM)")
    args = parser.parse_args()

    run_stage4(is_mock=args.mock)
