import sys
import os
import json
from pathlib import Path
from tqdm import tqdm

# Add the root directory to sys.path so we can import from src
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from src.config import SAMPLE_IDS, INPUT_DIR, DATA_DIR

STAGE5_DIR = DATA_DIR / "stage5_icd10"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def run_stage6():
    if not STAGE5_DIR.exists():
        print("Stage 5 output not found. Please run stage 5 first.")
        return

    # Process files
    for doc_id in tqdm(SAMPLE_IDS, desc="Processing Stage 6 Merge"):
        in_file = STAGE5_DIR / f"{doc_id}.json"
        out_file = OUTPUT_DIR / f"{doc_id}.json"
        source_file = INPUT_DIR / f"{doc_id}.txt"
        
        if not in_file.exists() or not source_file.exists():
            continue
            
        with open(source_file, "r", encoding="utf-8") as f:
            source_text = f.read()
            
        with open(in_file, "r", encoding="utf-8") as f:
            stage5_data = json.load(f)
            
        final_entities = []
        for ent in stage5_data:
            # 1. Validate text matches source at position
            start, end = ent["position"]
            actual_text = source_text[start:end]
            if actual_text != ent["text"]:
                print(f"Warning: Dropping entity '{ent['text']}' at {start}:{end} in {doc_id} because source has '{actual_text}'")
                continue
                
            # 2. Format candidates
            etype = ent.get("type", "")
            if "candidates" in ent:
                if etype in ["THUỐC", "CHẨN_ĐOÁN"]:
                    # Keep candidates even if empty
                    pass
                else:
                    # Remove candidates for other types
                    del ent["candidates"]
                    
            final_entities.append(ent)
            
        # 3. Sort chronologically by start position
        final_entities.sort(key=lambda x: x["position"][0])
            
        # Save as bare array
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(CompactPositionEncoder().encode(final_entities) + "\n")

if __name__ == "__main__":
    run_stage6()
