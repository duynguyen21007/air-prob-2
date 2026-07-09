import sys
import os
import json

from pathlib import Path
from tqdm import tqdm

# Add the root directory to sys.path so we can import from src
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from src.config import SAMPLE_IDS, INPUT_DIR, DATA_DIR
from src.retrieval.rxnorm_hybrid_search import RxNormHybridSearcher

STAGE3_DIR = DATA_DIR / "stage3_assertions"
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



def run_stage4():
    STAGE4_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    data_csv_path = BASE_DIR / "data_rxnorm.csv"
    chroma_persist_dir = DATA_DIR / "chroma_rxnorm_db"
    
    print("Initializing RxNorm Hybrid Searcher...")
    searcher = RxNormHybridSearcher(str(data_csv_path), str(chroma_persist_dir))
    
    for doc_id in tqdm(SAMPLE_IDS, desc="Processing Stage 4 RxNorm"):
        stage3_file = STAGE3_DIR / f"{doc_id}.json"
        out_file = STAGE4_OUT_DIR / f"{doc_id}.json"
        
        if not stage3_file.exists():
            print(f"Warning: Stage 3 output {stage3_file} does not exist. Skipping.")
            continue
            
        if out_file.exists():
            print(f"Skipping {doc_id} as it is already processed.")
            continue
            
        with open(stage3_file, "r", encoding="utf-8") as f:
            stage3_data = json.load(f)
            
        # Extract THUỐC entities
        thuoc_entities = [ent for ent in stage3_data if ent["type"] == "THUỐC"]
        
        lookup = {}
        if thuoc_entities:
            raw_drugs = [ent["text"] for ent in thuoc_entities]
            raw_drugs = list(set(raw_drugs))
            
            for drug in raw_drugs:
                rxcuis = searcher.get_qualified_rxcuis(drug, margin=0.05)
                if rxcuis:
                    lookup[drug] = rxcuis
                
        # Now rebuild entities with candidates
        entities = []
        for ent in stage3_data:
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
    run_stage4()
