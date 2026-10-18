# Hindi AI Content Quality Evaluation System

Minimal, flat-style Hindi NLP app for reviewing **grammatical correctness, semantic relevance, fluency, and cultural appropriateness**. Includes a Jupyter training notebook, open-data downloader, training script, FastAPI API, Streamlit dashboard, tests, annotation guidelines and a dashboard preview.

## Problem statement
Hindi AI-generated text can include grammar errors, awkward phrasing, off-topic content, or culturally insensitive assumptions. This project provides a repeatable review workflow with dimension estimates and surface-level flags.

## What is implemented
- Downloads a Hindi Wikipedia sample from [Hugging Face `wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia).
- Generates transparent provisional labels, trains TF-IDF character n-grams + multi-output logistic regression.
- Saves trained model and metrics to `models/`.
- FastAPI `/health`, `/model-info`, `/evaluate`.
- Streamlit flat/minimal dashboard with dimension cards, review flags, signals and JSON export.
- Notebook: `notebooks/hindi_content_quality_training.ipynb`.
- Human annotation guide and illustrative sample output.

## Methodology
1. Collect Hindi Wikipedia text.
2. Normalize and deduplicate.
3. Weak-label using simple, inspectable text heuristics.
4. Train a TF-IDF + logistic regression classifier for four dimensions.
5. Report hold-out macro-F1 and classification reports.
6. Serve model with FastAPI and render results in Streamlit.
7. Replace heuristic labels with diverse human ratings for research-grade evaluation.

**Critical limitation:** labels are weak heuristics, not expert judgments. Metrics only measure how well the model imitates the heuristics. Cultural appropriateness and semantic relevance are context-dependent; the starter model is not a validated quality judge.

## Quick start (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python download_data.py --limit 5000
python train.py
```
macOS/Linux: create/activate a venv with `python3 -m venv .venv` and `source .venv/bin/activate`, then run the same pip/download/train commands.

## Launch
Terminal 1: `uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000`
Terminal 2: `streamlit run app.py`
Dashboard: http://localhost:8501 · API docs: http://127.0.0.1:8000/docs

## Notebook training
Run `jupyter lab`, open `notebooks/hindi_content_quality_training.ipynb`, then execute cells top-to-bottom. It downloads data, builds labels, trains/evaluates the model, and saves artifacts.

## Sample API call
```bash
curl -X POST http://127.0.0.1:8000/evaluate -H "Content-Type: application/json" -d "{\"text\":\"यह एक उदाहरण वाक्य है।\"}"
```
See `docs/sample_outputs.md`. The sample is illustrative, not a result from a particular run.

## Structure
`app.py`, `download_data.py`, `train.py`, `backend/main.py`, `src/quality.py`, `notebooks/hindi_content_quality_training.ipynb`, `docs/`, `tests/`, `data/`, `models/`.

## Next steps
Use the included `docs/annotation_guidelines.md` to collect human annotations. Track annotator agreement, add original prompts for relevance judgments, evaluate genre/dialect slices, and document failure cases. Do not use this baseline for automatic censorship or high-impact decisions.
