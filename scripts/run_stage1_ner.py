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
from src.schema import NERResponseStage1
from src.prompts.stage1_ner import STAGE1_SYSTEM_PROMPT, STAGE1_USER_PROMPT_TEMPLATE
from src.postprocess import fix_position

STAGE1_OUT_DIR = DATA_DIR / "stage1_ner"

def run_stage1():
    STAGE1_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for doc_id in tqdm(SAMPLE_IDS, desc="Processing Stage 1 NER"):
        in_file = INPUT_DIR / f"{doc_id}.txt"
        out_file = STAGE1_OUT_DIR / f"{doc_id}.json"
        
        if not in_file.exists():
            print(f"Warning: Input file {in_file} does not exist. Skipping.")
            continue
            
        if out_file.exists():
            print(f"Skipping {doc_id} as it is already processed.")
            continue
            
        with open(in_file, "r", encoding="utf-8") as f:
            text = f.read()
            
        prompt = STAGE1_USER_PROMPT_TEMPLATE.format(text=text)
        
        try:
            # Call Gemini
            parsed_response = generate_structured_response(
                prompt=prompt,
                response_schema=NERResponseStage1,
                system_instruction=STAGE1_SYSTEM_PROMPT
            )
            
            # Post-process: fix positions
            entities = []
            seen = set()
            for ent in parsed_response.entities:
                fixed_pos = fix_position(text, ent.text, ent.position)
                key = (ent.text, tuple(fixed_pos))
                if key in seen:
                    continue
                seen.add(key)
                entities.append({
                    "text": ent.text,
                    "position": fixed_pos
                })
                
            # Save result
            result = {"entities": entities}
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Error processing document {doc_id}: {e}")

if __name__ == "__main__":
    run_stage1()
