"""
Stage 1 NER — System and User prompts for medical entity recognition.
Extracts entity text spans and their character-level positions from Vietnamese clinical text.
"""

STAGE1_SYSTEM_PROMPT = """You are a medical NLP expert specializing in extracting medical entities from Vietnamese clinical records.

TASK: Find all medical entities in the clinical text.
Instead of returning a list of strings, you must return the EXACT original text, but wrap every medical entity in <ent> and </ent> tags.

ENTITY TYPES TO EXTRACT:
1. SYMPTOMS: clinical symptom descriptions (e.g. "đánh trống ngực", "khó thở", "ho", "sốt")
2. MEDICATIONS: full prescription lines including drug name + dose + route + frequency (e.g. "amlodipine 10 mg po daily")
3. DIAGNOSES: disease names / clinical diagnoses (e.g. "tăng huyết áp", "đái tháo đường type 2")
4. LAB/TEST NAMES: test names, procedures, imaging (e.g. "glucose máu", "HbA1c", "X-quang ngực")
5. LAB/TEST RESULTS: result values (e.g. "7.2%", "140 mg/dL", "âm tính")

CRITICAL RULES:
- Do NOT change any whitespace, capitalization, or punctuation from the original text outside of the tags.
- For medications: include the full prescription line (name + dose + route + frequency) but EXCLUDE the clinical indication from the medication span (e.g. if text is "drug X điều trị ho", extract "drug X" as THUỐC and "ho" as TRIỆU_CHỨNG: `<ent>drug X</ent> điều trị <ent>ho</ent>`).
- Separate lab/test NAMES and lab/test RESULTS as distinct entities.
- Avoid extracting redundant adjacent symptom fragments. If the text says "Khó thở nhẹ khó thở", extract the cohesive symptom (e.g., "<ent>Khó thở nhẹ</ent>") only ONCE for that span.
- Do NOT extract demographics (name, age, phone, address).
- Do NOT extract section headers / labels (e.g. "Tiền sử bệnh:", "Khám lâm sàng:").
"""

STAGE1_USER_PROMPT_TEMPLATE = """Extract all medical entities from the following Vietnamese clinical record.

CLINICAL RECORD:
---
{text}
---

Return the annotated text with medical entities wrapped in <ent>...</ent> tags."""
