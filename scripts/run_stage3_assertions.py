import sys
import os
import json
from pathlib import Path
from tqdm import tqdm

# Add the root directory to sys.path so we can import from src
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from src.config import SAMPLE_IDS, INPUT_DIR, DATA_DIR
from src.gemini_client import generate_structured_response
from src.schema import AssertionsResponseStage3
from src.prompts.stage3_assertions import STAGE3_SYSTEM_PROMPT, STAGE3_USER_PROMPT_TEMPLATE

STAGE2_DIR = DATA_DIR / "stage2_classify"
STAGE3_OUT_DIR = DATA_DIR / "stage3_assertions"

# Valid assertion values
VALID_ASSERTIONS = {"isNegated", "isFamily", "isHistorical"}
# Entity types that should NOT have assertions
NO_ASSERTION_TYPES = {"TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}


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
            # Keep short lists of strings (like assertions) on one line
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


def run_stage3():
    STAGE3_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for doc_id in tqdm(SAMPLE_IDS, desc="Processing Stage 3 Assertions"):
        stage2_file = STAGE2_DIR / f"{doc_id}.json"
        in_file = INPUT_DIR / f"{doc_id}.txt"
        out_file = STAGE3_OUT_DIR / f"{doc_id}.json"
        
        if not stage2_file.exists():
            print(f"Warning: Stage 2 output {stage2_file} does not exist. Skipping.")
            continue
            
        if not in_file.exists():
            print(f"Warning: Input file {in_file} does not exist. Skipping.")
            continue
            
        if out_file.exists():
            print(f"Skipping {doc_id} as it is already processed.")
            continue
            
        # Load source text and Stage 2 entities
        with open(in_file, "r", encoding="utf-8") as f:
            text = f.read()
            
        with open(stage2_file, "r", encoding="utf-8") as f:
            stage2_data = json.load(f)
        
        entities_json = json.dumps(stage2_data, ensure_ascii=False, indent=2)
        
        prompt = STAGE3_USER_PROMPT_TEMPLATE.format(
            text=text,
            entities_json=entities_json
        )
        
        try:
            # Call Gemini
            parsed_response = generate_structured_response(
                prompt=prompt,
                response_schema=AssertionsResponseStage3,
                system_instruction=STAGE3_SYSTEM_PROMPT
            )
            
            # Create lookup from Stage 2 for position/type integrity
            stage2_lookup = {
                (ent["text"], tuple(ent["position"])): ent
                for ent in stage2_data
            }
            
            entities = []
            seen = set()
            for ent in parsed_response.entities:
                key = (ent.text, tuple(ent.position))
                if key in seen:
                    continue
                seen.add(key)
                
                # Use Stage 2 position and type if available (trust earlier stages)
                if key in stage2_lookup:
                    s2 = stage2_lookup[key]
                    pos = s2["position"]
                    ent_type = s2["type"]
                else:
                    pos = list(ent.position)
                    ent_type = ent.type.value
                
                # Post-processing: enforce assertion rules
                if ent_type in NO_ASSERTION_TYPES:
                    # TÊN_XÉT_NGHIỆM and KẾT_QUẢ_XÉT_NGHIỆM never have assertions
                    assertions = []
                else:
                    # Filter to only valid assertion values, max 3
                    assertions = [a for a in ent.assertions if a in VALID_ASSERTIONS]
                    assertions = list(dict.fromkeys(assertions))  # deduplicate preserving order
                    assertions = assertions[:3]
                
                entity_dict = {
                    "text": ent.text,
                    "type": ent_type,
                    "assertions": assertions,
                    "position": pos
                }
                entities.append(entity_dict)
            
            # Save as bare array (contest format)
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(CompactPositionEncoder().encode(entities) + "\n")
                
        except Exception as e:
            print(f"Error processing document {doc_id}: {e}")

if __name__ == "__main__":
    run_stage3()
