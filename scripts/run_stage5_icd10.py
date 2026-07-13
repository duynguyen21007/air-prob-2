import sys
import os
import json
from pathlib import Path
from tqdm import tqdm

# Add the root directory to sys.path so we can import from src
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from src.config import SAMPLE_IDS, INPUT_DIR, DATA_DIR
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


def run_stage5():
    if not STAGE4_DIR.exists():
        print("Stage 4 output not found. Please run stage 4 first.")
        return

    data_csv_path = BASE_DIR / "data_icds.csv"
    chroma_persist_dir = DATA_DIR / "chroma_icd10_db"
    
    if not chroma_persist_dir.exists():
        print("ChromaDB not found! Please run `python scripts/build_icd10_index.py` first.")
        return
        
    print("Initializing Icd10HybridSearcher...")
    searcher = Icd10HybridSearcher(str(data_csv_path), str(chroma_persist_dir))

    json_files = list(STAGE4_DIR.glob("*.json"))
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
                
        # 2. Get ICD-10 candidates that pass reranker qualification rules
        lookup = {}
        if diagnoses:
            for diag in diagnoses:
                qualified_icds = searcher.get_qualified_icds(
                    diag,
                    margin=0.05,
                    absolute_threshold=0.5,
                )
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
    run_stage5()
