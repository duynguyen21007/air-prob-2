import sys
import os
import json
import time
import requests
from pathlib import Path
from tqdm import tqdm

# Add the root directory to sys.path so we can import from src
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from src.config import SAMPLE_IDS, INPUT_DIR, DATA_DIR
from src.gemini_client import generate_structured_response
from src.schema import RxNormCleanResponse
from src.prompts.stage4_rxnorm import STAGE4_SYSTEM_PROMPT, STAGE4_USER_PROMPT_TEMPLATE

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


def get_rxnorm_id(clean_name):
    # Call RxNorm REST API
    url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={clean_name}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        id_group = data.get("idGroup", {})
        rxnorm_ids = id_group.get("rxnormId", [])
        if rxnorm_ids:
            return rxnorm_ids[0]
    except Exception as e:
        print(f"RxNorm API error for '{clean_name}': {e}")
    return None


def run_stage4():
    STAGE4_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
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
            # Deduplicate for prompt
            raw_drugs = list(set(raw_drugs))
            
            prompt = STAGE4_USER_PROMPT_TEMPLATE.format(
                drugs_json=json.dumps(raw_drugs, ensure_ascii=False, indent=2)
            )
            
            try:
                parsed_response = generate_structured_response(
                    prompt=prompt,
                    response_schema=RxNormCleanResponse,
                    system_instruction=STAGE4_SYSTEM_PROMPT
                )
                for item in parsed_response.drugs:
                    lookup[item.original_text] = item.clean_name
            except Exception as e:
                print(f"Error calling Gemini for document {doc_id}: {e}")
                
        # Now rebuild entities with candidates
        entities = []
        for ent in stage3_data:
            new_ent = dict(ent) # Make a copy
            if new_ent["type"] == "THUỐC":
                clean_name = lookup.get(new_ent["text"])
                if clean_name:
                    rxcui = get_rxnorm_id(clean_name)
                    if rxcui:
                        new_ent["candidates"] = [rxcui]
                    else:
                        new_ent["candidates"] = []
                else:
                    new_ent["candidates"] = []
                time.sleep(0.1) # Be nice to RxNorm API
            
            entities.append(new_ent)
            
        # Save as bare array (contest format)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(CompactPositionEncoder().encode(entities) + "\n")

if __name__ == "__main__":
    run_stage4()
