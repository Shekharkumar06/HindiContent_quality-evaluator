---
title: Hindi Content Quality Evaluator
emoji: ◈
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.50.0
app_file: app.py
pinned: false
---

# Hindi AI Content Quality Evaluator

A Streamlit dashboard for experimental Hindi AI-content quality review across four dimensions: grammatical correctness, semantic relevance, fluency, and cultural appropriateness. It also exposes inspectable text signals and a downloadable JSON result.

## Hugging Face Spaces

This branch (`hf-deploy`) is prepared for a Streamlit Space:

1. Create a new **Streamlit** Space on Hugging Face.
2. Upload/push the contents of this branch to that Space's repository.
3. Hugging Face reads the Space metadata above and installs `requirements.txt`.
4. The Space runs `app.py`; no separate FastAPI service is needed.

The Space uses repository-relative paths. The serialized model artifact is not bundled in this lightweight deployment branch. The dashboard therefore runs with the transparent heuristic baseline, and it will automatically use `models/quality_model.joblib` when a compatible trained artifact is present. Training data and heavyweight training-only dependencies are not installed in the Space.

## Features

- Four dimension estimates: grammar, relevance, fluency, and cultural context.
- Surface-level review flags and text signals.
- JSON output and downloadable evaluation result.
- Lightweight fallback when the optional model artifact is missing or cannot be loaded.
- Original training workflow, FastAPI service, notebook, evaluation metrics, and annotation documentation remain in the repository.

## Local run

Use Python 3.11 or 3.12 in a virtual environment:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The original FastAPI backend is still available for local/API deployments. Launch it separately with `uvicorn backend.main:app --reload --port 8000`; it expects the trained artifact under `models/quality_model.joblib`.

## Training (optional, local)

The training and download workflow has extra dependencies not installed by the Space's minimal runtime. Install the training dependencies separately, run `python download_data.py --limit 5000`, then `python train.py` to generate model artifacts. Training labels are weak heuristics, not human judgments.

## Limitations

- This is a research/demo baseline, not a validated content-quality judge.
- Semantic relevance cannot be reliably assessed without the original prompt/context.
- Cultural appropriateness is context-sensitive and should receive human review.
- No flagged signal does not establish that content is correct or safe.

See `docs/annotation_guidelines.md` for human-evaluation guidance.
