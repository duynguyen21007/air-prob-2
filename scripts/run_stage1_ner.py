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


STAGE1_OUT_DIR = DATA_DIR / "stage1_ner"


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
                items.append(indent * (indent_level + 1) + self._encode(item, indent_level + 1))
            return "[\n" + ",\n".join(items) + "\n" + indent * indent_level + "]"
        else:
            return json.dumps(o, ensure_ascii=False)


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
            source_text = f.read()
            
        prompt = STAGE1_USER_PROMPT_TEMPLATE.format(text=source_text)
        
        try:
            # Call Gemini
            parsed_response = generate_structured_response(
                prompt=prompt,
                response_schema=NERResponseStage1,
                system_instruction=STAGE1_SYSTEM_PROMPT
            )
            
            # Extract entities using regex
            import re
            annotated_text = parsed_response.annotated_text
            matches = re.findall(r'<ent>(.*?)</ent>', annotated_text)
            
            entities = []
            seen = set()
            
            for entity_text in matches:
                # 1. Deduplicate by lowercased text
                key = entity_text.lower()
                if key in seen:
                    continue
                    
                # 2. Find exact position in source text
                idx = source_text.find(entity_text)
                if idx != -1:
                    seen.add(key)
                    entities.append({
                        "text": entity_text,
                        "position": [idx, idx + len(entity_text)]
                    })
                else:
                    # If LLM slightly hallucinated characters, it won't match exactly.
                    # We skip it because contest requires exact substrings.
                    print(f"Warning: Entity '{entity_text}' not found in source text, skipping.")
            
            # Format output dictionary
            output_data = {"entities": entities}
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(CompactPositionEncoder().encode(output_data) + "\n")
                
        except Exception as e:
            print(f"Error processing document {doc_id}: {e}")

if __name__ == "__main__":
    run_stage1()
