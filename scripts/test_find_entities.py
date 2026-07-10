import json, os, time, traceback
from tqdm import tqdm
from openai import OpenAI

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
    "4": "đã tiến hành tổng phân tích tế bào máu bằng máy lazer (tbm): WBC:14,43; NEUT% (Tỷ lệ % bạch cầu trung tính):76,4; LYPH% (Tỷ lệ bạch cầu lympho):12,8;""
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


OPENAI_API_KEY="dummy"
client = OpenAI(api_key=OPENAI_API_KEY, base_url="http://localhost:8211/v1")
llm_config = {
    #"model": "Qwen/Qwen3.6-27B-FP8",
    "model": "Qwen/Qwen3.5-9B",
    "max_token": 4096,
    "temperature": 0.0,
}

def get_response_for_single_chat(prompt):
    start = time.time()
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        response = client.chat.completions.create(
            model=llm_config['model'],
            temperature=llm_config['temperature'],
            max_tokens=llm_config['max_token'],
            messages = messages,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            # reasoning_effort="low"
        )
        # print(response)
        response = response.choices[0].message.content
    except Exception as e:
        raise Exception(e)
    return response

def postprocess(response, dict_lines, dict_spans):
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
            p = response[start_pos + len(f"<{key}>") : end_pos]
            p = p.strip()
            print('p =', p)
            p = json.loads(p)
            if not isinstance(p, dict):
                print('fail1')
                raise ValueError()
            result[key] = []
            for x in p:
                print('x= ', x)
                if str(x) not in dict_lines:
                    continue
                for text in p[x]:
                    if text not in dict_lines[str(x)]:
                        continue
                    print(text)
                    position = dict_lines[str(x)].index(text)
                    offset = dict_spans[str(x)][0]
                    print(offset, position)
                    result[key].append({"text": text, "type": key, "position": [offset + position, offset + position + len(text)], "assertions": [], "candidates": [] })
        except Exception as e:
            print('fail4', e)
            print(traceback.format_exc())
            result[key] = []
            continue
    new_result = []
    for key in result:
        new_result.extend(result[key])
    result = sorted(new_result, key=lambda x: x["position"][0])
    return result

def process(text):
    dict_lines = {}
    dict_spans = {}

    pos = 0
    i = 1

    for line in text.split('\n'):
        line_start = pos
        line_end = line_start + len(line)

        stripped = line.strip()
        if stripped:
            # vị trí sau khi strip
            left = len(line) - len(line.lstrip())
            right = len(line.rstrip())

            start = line_start + left
            end = line_start + right

            dict_lines[str(i)] = stripped
            dict_spans[str(i)] = (start, end)   # end là exclusive
            i += 1

        # +1 vì ký tự '\n'
        pos = line_end + 1

    print( json.dumps(dict_lines, indent=4, ensure_ascii=False) ) 
    prompt = prompt_template.format(text=json.dumps(dict_lines, indent=4, ensure_ascii=False))
    response = get_response_for_single_chat(prompt)
    result = postprocess(response, dict_lines, dict_spans)
    return result


folder = "find_entities"
os.makedirs(folder, exist_ok=True)
for file in sorted([int(file[:-4]) for file in os.listdir("input") if file.endswith('.txt')]):
    print('*' * 20)
    print(file)
    text = open(f"input/{file}.txt", 'r' ).read()
    result = process(text)
    print(result)
    with open(f"{folder}/{file}.json", 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
