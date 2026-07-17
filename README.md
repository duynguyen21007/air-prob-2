# Viettel AI Race 2026 — Pipeline

Staged LLM pipeline for clinical entity extraction using vLLM + hybrid retrieval.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start vLLM server (Docker)

```bash
docker run -d --name vllm-qwen3.5-9b --gpus '"device=6"' -p 8211:8000 -v $(pwd)/model:/root/.cache/huggingface --ipc=host vllm/vllm-openai:v0.21.0-cu129 --model Qwen/Qwen3.5-9B --dtype bfloat16 --gpu-memory-utilization 0.4 --trust-remote-code --max-model-len 16000 --max-num-seqs 1
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` to match your vLLM server settings (defaults point to `http://localhost:8211/v1`).

> **Mock / Offline Mode**: If you do not have a GPU or vLLM server running, set `MOCK_LLM=true` in `.env`. The pipeline will automatically use pre-saved responses in `mock_data/stage1_ner/` and local vector search reranking to run end-to-end.

### 4. Build retrieval indexes (one-time)

```bash
python scripts/build_icd10_index.py
python scripts/build_rxnorm_index.py
```

### 5. Run pipeline

```bash
python run_pipeline.py
```

## Pipeline Stages

| Stage | Task | Script | Output |
|-------|------|--------|--------|
| 1 | NER — extract entity text + position + type via vLLM | `run_stage1_ner.py` | `data/stage1_ner/` |
| 2 | Classify — assign entity type (skipped, done in Stage 1) | `run_stage2_classify.py` | — |
| 3 | Assertions — negation/family/historical (skipped, set to []) | `run_stage3_assertions.py` | — |
| 4 | RxNorm — drug code linking (hybrid BM25 + dense + LLM reranking) | `run_stage4_rxnorm.py` | `data/stage4_rxnorm/` |
| 5 | ICD-10 — diagnosis code linking (hybrid BM25 + dense + LLM reranking) | `run_stage5_icd10.py` | `data/stage5_icd10/` |
| 6 | Merge — filter, sort, output final contest JSON | `run_stage6_merge.py` | `output/` |

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
        "assertions": [],
        "position": [53, 75]
    }
]
```

## Project Layout

```
input/          # Source clinical notes (100 files)
data/           # Per-stage pipeline outputs (ignored by git)
mock_data/      # Pre-saved mock Stage 1 responses (tracked in git)
output/         # Final contest JSON (Stage 6)
src/            # Shared code (config, LLM client, retrieval, schemas)
scripts/        # Runner scripts + index builders
config.yaml     # Sample IDs
.env            # vLLM server and mock mode configuration
```
