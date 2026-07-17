import sys
import os
import json
import shutil
import traceback
from pathlib import Path
from tqdm import tqdm

# Add the root directory to sys.path so we can import from src
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from src.config import SAMPLE_IDS, INPUT_DIR, DATA_DIR, MOCK_DATA_DIR
from src.llm_client import get_response_for_single_chat


STAGE1_OUT_DIR = DATA_DIR / "stage1_ner"
MOCK_STAGE1_DIR = MOCK_DATA_DIR / "stage1_ner"

prompt_template = """You are a medical NLP expert specializing in extracting medical entities from Vietnamese clinical records.

# TASK: Find all medical entities in the clinical text.
Instead of returning a list of strings, you must return the EXACT original text, but wrap every medical entity in <ent> and </ent> tags.

5 ENTITY TYPES:
1. TRIỆU_CHỨNG: Tên triệu chứng bệnh nhân mắc phải
   Examples: "đánh trống ngực", "khó thở", "ho", "đau ngực", "buồn nôn", "sốt"
   Includes negated symptoms ("Không buồn nôn") — still TRIỆU_CHỨNG

2. THUỐC: Tên thuốc mà bệnh nhân điều trị
   Examples: "metoprolol 25mg po bid", "amlodipine 10 mg po daily", "aspirin"
   Includes historical meds, current meds, and newly prescribed meds — all are THUỐC

3. CHẨN_ĐOÁN: Tên chẩn đoán của bác sĩ về bệnh mà bệnh nhân mắc phải
   Examples: "tăng huyết áp", "đái tháo đường type 2", "xơ gan do rượu"
   Includes disease names from "tiền sử" (history) or "chẩn đoán" (diagnosis) sections

4. TÊN_XÉT_NGHIỆM: Tên xét nghiệm bệnh nhân thực hiện
   Examples: "troponin", "HbA1c", "chụp x-quang ngực", "điện tâm đồ (ecg)", "siêu âm", "glucose máu"
   Includes test group names ("bảng công thức máu") and procedures ("sinh thiết")

5. KẾT_QUẢ_XÉT_NGHIỆM: Kết quả xét nghiệm bệnh nhân thực hiện, bao gồm giá trị và đơn vị của xét nghiệm
   Examples: "7.2%", "140 mg/dL", "âm tính", "bình thường", "0.01", "94-95 RA", "159/72", "tim to"
   Includes vital sign values (pulse, blood pressure, SpO2)

text is dict, contains many sentences, each sentence has a sentence id. Each entity outputs a dict, each key is a sentence id, each value is a list of extracted entities in that sentence.

# Example:
text:
{{
    "1": Bệnh nhân nam 70 tuổi bị bệnh 1 tuần nay, ho đờm xanh, tức ngực, đau thượng vị, ợ hơi, được chẩn đoán mắc bệnh trào ngược dạ dày - thực quản.",
    "2": "Bệnh nhân có tiền sử sử dụng Chlorpheniramine 0.4 MG/ML", 
    "3": "Capsaicin 0.38 MG/ML",
    "4": "đã tiến hành tổng phân tích tế bào máu bằng máy lazer (tbm): WBC:14,43; NEUT% (Tỷ lệ % bạch cầu trung tính):76,4; LYPH% (Tỷ lệ bạch cầu lympho):12,8;"
}}

output:
<CHẨN_ĐOÁN>
{{
    "1": ["bệnh trào ngược dạ dày - thực quản"]
}}
</CHẨN_ĐOÁN>

<TRIỆU_CHỨNG>
{{
    "1": ["ho đờm xanh", "tức ngực", "đau thượng vị", "ợ hơi"]
}}
</TRIỆU_CHỨNG>

<TÊN_XÉT_NGHIỆM>
{{
    "4": ["WBC", "NEUT% (Tỷ lệ % bạch cầu trung tính)", "LYPH% (Tỷ lệ bạch cầu lympho)"]
}}
</TÊN_XÉT_NGHIỆM>

<KẾT_QUẢ_XÉT_NGHIỆM>
{{
    "4": ["14,43", "76,4", "12,8"]
}}
</KẾT_QUẢ_XÉT_NGHIỆM>

<THUỐC>
{{
    "2": ["Chlorpheniramine 0.4 MG/ML"], 
    "3": ["Capsaicin 0.38 MG/ML"]
}}
</THUỐC>

---
text:
"{text}"

output:
"""


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


def postprocess_ner(response, dict_lines, dict_spans):
    """Parse LLM NER response with XML-tagged entity types into structured entities."""
    keys = ["CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM", "THUỐC"]
    result = {}
    for key in keys:
        if f"<{key}>" not in response or f"</{key}>" not in response:
            result[key] = []
            continue
        try:
            start_pos = response.find(f"<{key}>")
            end_pos = response.find(f"</{key}>")
            if end_pos < start_pos:
                raise ValueError()    
            p = response[start_pos + len(f"<{key}>"): end_pos]
            p = p.strip()
            p = json.loads(p)
            if not isinstance(p, dict):
                raise ValueError()
            result[key] = []
            for x in p:
                if str(x) not in dict_lines:
                    continue
                for text in p[x]:
                    if text not in dict_lines[str(x)]:
                        continue
                    position = dict_lines[str(x)].index(text)
                    offset = dict_spans[str(x)][0]
                    result[key].append({
                        "text": text,
                        "type": key,
                        "position": [offset + position, offset + position + len(text)],
                        "assertions": [],
                        "candidates": []
                    })
        except Exception as e:
            print(f"Error parsing entity type {key}: {e}")
            print(traceback.format_exc())
            result[key] = []
            continue

    new_result = []
    for key in result:
        new_result.extend(result[key])
    result = sorted(new_result, key=lambda x: x["position"][0])
    return result


def process_text(text):
    """Split text into numbered lines and call vLLM for NER extraction."""
    dict_lines = {}
    dict_spans = {}

    pos = 0
    i = 1

    for line in text.split('\n'):
        line_start = pos
        line_end = line_start + len(line)

        stripped = line.strip()
        if stripped:
            # Position after stripping whitespace
            left = len(line) - len(line.lstrip())
            right = len(line.rstrip())

            start = line_start + left
            end = line_start + right

            dict_lines[str(i)] = stripped
            dict_spans[str(i)] = (start, end)   # end is exclusive
            i += 1

        # +1 for the '\n' character
        pos = line_end + 1

    prompt = prompt_template.format(text=json.dumps(dict_lines, indent=4, ensure_ascii=False))
    response = get_response_for_single_chat(prompt)
    result = postprocess_ner(response, dict_lines, dict_spans)
    return result


def run_stage1():
    STAGE1_OUT_DIR.mkdir(parents=True, exist_ok=True)
    mock_mode = os.getenv("MOCK_LLM", "false").lower() in ("true", "1", "yes")
    
    for doc_id in tqdm(SAMPLE_IDS, desc="Processing Stage 1 NER"):
        in_file = INPUT_DIR / f"{doc_id}.txt"
        out_file = STAGE1_OUT_DIR / f"{doc_id}.json"
        mock_file = MOCK_STAGE1_DIR / f"{doc_id}.json"
        
        if not in_file.exists():
            print(f"Warning: Input file {in_file} does not exist. Skipping.")
            continue
            
        if out_file.exists():
            print(f"Skipping {doc_id} as it is already processed.")
            continue
            
        # If in mock mode or mock_file exists and out_file doesn't, copy mock file
        if mock_file.exists() and mock_mode:
            print(f"Using pre-saved mock response for document {doc_id}.")
            shutil.copy(mock_file, out_file)
            continue

        with open(in_file, "r", encoding="utf-8") as f:
            source_text = f.read()
            
        try:
            result = process_text(source_text)
            
            # Save as bare array (contest format)
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(CompactPositionEncoder().encode(result) + "\n")
                
        except Exception as e:
            # Fallback to mock file if available upon LLM error
            if mock_file.exists():
                print(f"LLM error for document {doc_id}. Falling back to pre-saved mock data.")
                shutil.copy(mock_file, out_file)
            else:
                print(f"Error processing document {doc_id}: {e}")


if __name__ == "__main__":
    run_stage1()
