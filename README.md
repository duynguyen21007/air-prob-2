# Viettel AI Race 2026 — Baseline Pipeline

Staged LLM pipeline for clinical entity extraction using Google Gemini.

## Stage 0 — Setup

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

`config.yaml` controls the 12-file sample set and model settings:

```yaml
sample_ids: [1, 2, 3, 5, 10, 15, 20, 30, 50, 60, 80, 100]
model: gemini-2.5-flash
temperature: 0.1
```

## Project layout

```
input/          # Source clinical notes (100 files)
data/           # Per-stage pipeline outputs (created as stages run)
output/         # Final contest JSON (Stage 6)
src/            # Shared code (config, Gemini client, prompts, …)
scripts/        # One runner script per stage (added incrementally)
```

## Development protocol

Each pipeline stage is implemented and reviewed separately. After Stage 1 is ready, run:

```bash
python scripts/run_stage1_ner.py
python scripts/review_stage.py --stage 1 --summary
```

Then review output in `data/stage1_ner/` before proceeding to Stage 2.
