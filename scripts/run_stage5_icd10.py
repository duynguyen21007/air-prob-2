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
from src.schema import ICD10CleanResponse
from src.prompts.stage5_icd10 import STAGE5_SYSTEM_PROMPT, STAGE5_USER_PROMPT_TEMPLATE

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

    # Process files
    for doc_id in tqdm(SAMPLE_IDS, desc="Processing Stage 5 ICD-10"):
        in_file = STAGE4_DIR / f"{doc_id}.json"
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
                
        # 2. Get clean ICD-10 from Gemini
        lookup = {}
        if diagnoses:
            diagnoses_list = list(diagnoses)
            # Batch them into chunks if needed, but usually 10-20 diagnoses fit easily in one prompt
            prompt = STAGE5_USER_PROMPT_TEMPLATE.format(
                diagnoses_json=json.dumps(diagnoses_list, ensure_ascii=False, indent=2)
            )
            
            try:
                parsed_response = generate_structured_response(
                    prompt=prompt,
                    response_schema=ICD10CleanResponse,
                    system_instruction=STAGE5_SYSTEM_PROMPT
                )
                for item in parsed_response.diagnoses:
                    lookup[item.original_text] = item.icd10_code
            except Exception as e:
                print(f"Error calling Gemini for document {doc_id}: {e}")
                
        # 3. Now rebuild entities with candidates
        entities = []
        for ent in stage4_data:
            new_ent = dict(ent) # Make a copy
            if new_ent["type"] == "CHẨN_ĐOÁN":
                icd_code = lookup.get(new_ent["text"])
                if icd_code:
                    new_ent["candidates"] = [icd_code]
                else:
                    new_ent["candidates"] = []
            
            entities.append(new_ent)
            
        # Save as bare array (contest format)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(CompactPositionEncoder().encode(entities) + "\n")

if __name__ == "__main__":
    run_stage5()
