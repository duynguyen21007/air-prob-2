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
from src.schema import ClassifyResponseStage2
from src.prompts.stage2_classify import STAGE2_SYSTEM_PROMPT, STAGE2_USER_PROMPT_TEMPLATE

STAGE1_DIR = DATA_DIR / "stage1_ner"
STAGE2_OUT_DIR = DATA_DIR / "stage2_classify"

def run_stage2():
    STAGE2_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for doc_id in tqdm(SAMPLE_IDS, desc="Processing Stage 2 Classify"):
        stage1_file = STAGE1_DIR / f"{doc_id}.json"
        in_file = INPUT_DIR / f"{doc_id}.txt"
        out_file = STAGE2_OUT_DIR / f"{doc_id}.json"
        
        if not stage1_file.exists():
            print(f"Warning: Stage 1 output {stage1_file} does not exist. Skipping.")
            continue
            
        if not in_file.exists():
            print(f"Warning: Input file {in_file} does not exist. Skipping.")
            continue
            
        if out_file.exists():
            print(f"Skipping {doc_id} as it is already processed.")
            continue
            
        # Load source text and Stage 1 entities
        with open(in_file, "r", encoding="utf-8") as f:
            text = f.read()
            
        with open(stage1_file, "r", encoding="utf-8") as f:
            stage1_data = json.load(f)
        
        entities_json = json.dumps(stage1_data["entities"], ensure_ascii=False, indent=2)
        
        prompt = STAGE2_USER_PROMPT_TEMPLATE.format(
            text=text,
            entities_json=entities_json
        )
        
        try:
            # Call Gemini
            parsed_response = generate_structured_response(
                prompt=prompt,
                response_schema=ClassifyResponseStage2,
                system_instruction=STAGE2_SYSTEM_PROMPT
            )
            
            # Build output — keep text/position from Stage 1, add type from Stage 2
            # Create a lookup from Stage 1 for position integrity
            stage1_lookup = {
                (ent["text"], tuple(ent["position"])): ent
                for ent in stage1_data["entities"]
            }
            
            entities = []
            seen = set()
            for ent in parsed_response.entities:
                key = (ent.text, tuple(ent.position))
                if key in seen:
                    continue
                seen.add(key)
                
                # Use Stage 1 position if available (trust Stage 1 positions)
                if key in stage1_lookup:
                    pos = stage1_lookup[key]["position"]
                else:
                    pos = list(ent.position)
                    
                # Build entity dict matching contest output field order
                entity_dict = {
                    "text": ent.text,
                    "type": ent.type.value,
                    "assertions": [],
                    "position": pos
                }
                entities.append(entity_dict)
            
            # Save as bare array (contest format)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(entities, f, ensure_ascii=False, indent=4)
                
        except Exception as e:
            print(f"Error processing document {doc_id}: {e}")

if __name__ == "__main__":
    run_stage2()
