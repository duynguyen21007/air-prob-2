# Viettel AI Race 2026 — Baseline Pipeline

Staged LLM pipeline for clinical entity extraction using Google Gemini.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
```

Edit `.env` and set your `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey).

### 3. Verify installation

```bash
python -c "from src.gemini_client import client; print('OK')"
```

This checks that the project scaffold and Gemini client import correctly. If `GEMINI_API_KEY` is set, the client is ready for API calls in later stages.

### Configuration

`config.yaml` controls sample set and model settings:

```yaml
sample_ids: [0, 1, 2, 3, 5]
model: gemini-3.1-flash-lite
temperature: 0.1
```

## Pipeline Stages

| Stage | Task | Script | Output |
|-------|------|--------|--------|
| 1 | NER — extract entity text + position | `run_stage1_ner.py` | `data/stage1_ner/` |
| 2 | Classify — assign entity type | `run_stage2_classify.py` | `data/stage2_classify/` |
| 3 | Assertions — negation/family/historical | `run_stage3_assertions.py` | `data/stage3_assertions/` |
| 4 | RxNorm — drug code linking | *coming soon* | `data/stage4_rxnorm/` |
| 5 | ICD-10 — diagnosis code linking | *coming soon* | `data/stage5_icd10/` |
| 6 | Merge — final contest JSON | *coming soon* | `output/` |

## Usage

```bash
# Stage 1: Extract entities
python scripts/run_stage1_ner.py

# Stage 2: Classify entity types
python scripts/run_stage2_classify.py

# Stage 3: Detect assertions (negation/family/historical)
python scripts/run_stage3_assertions.py

# Review any stage
python scripts/review_stage.py --stage 1 --id 1
python scripts/review_stage.py --stage 2 --summary
python scripts/review_stage.py --stage 3 --id 1
```

## Entity Types

| Type | Description |
|------|-------------|
| `TRIỆU_CHỨNG` | Clinical symptoms |
| `THUỐC` | Medications (name + dose + route + frequency) |
| `CHẨN_ĐOÁN` | Diagnoses / conditions |
| `TÊN_XÉT_NGHIỆM` | Lab/test names, procedures, imaging |
| `KẾT_QUẢ_XÉT_NGHIỆM` | Lab values, test results |

## Output Format

Follows contest specification — bare JSON array per file:

```json
[
    {
        "text": "metoprolol 25mg po bid",
        "type": "THUỐC",
        "candidates": ["866924"],
        "assertions": ["isHistorical"],
        "position": [53, 75]
    }
]
```

## Project Layout

```
input/          # Source clinical notes (100 files)
data/           # Per-stage pipeline outputs
output/         # Final contest JSON (Stage 6)
src/            # Shared code (config, Gemini client, schemas, prompts)
scripts/        # Runner + review scripts
config.yaml     # Sample IDs, model, temperature
```
