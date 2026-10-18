import json
from pathlib import Path

import joblib
import streamlit as st

from src.quality import DIMS, normalize_text, text_signals, weak_labels, score_to_percent

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "quality_model.joblib"

st.set_page_config(page_title="Hindi Content Quality", page_icon="◈", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1120px; padding-top: 2rem}
h1,h2 {letter-spacing: -0.04em}
.hero,.metric {padding: 1rem 1.2rem; border: 1px solid #e5e7eb; border-radius: 14px; background: #fff}
.eyebrow,.muted {color: #64748b}
.eyebrow {font-size: .75rem; letter-spacing: .12em; font-weight: 700}
.metric-label {font-size: .82rem; color: #64748b}
.metric-value {font-size: 1.8rem; font-weight: 750; color: #111827}
div.stButton>button {background: #111827; color: white; border-radius: 10px; border: 0}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the optional trained artifact once per Streamlit process."""
    if not MODEL_PATH.is_file():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        return exc


def evaluate_text(text: str, model):
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("Text cannot be empty.")

    signals = text_signals(normalized)
    flags = []
    if signals["token_count"] < 4:
        flags.append("Text is very short; estimates may be unreliable.")
    if signals["repeated_punctuation"]:
        flags.append("Repeated punctuation detected.")
    if signals["repeated_token_count"] >= 3:
        flags.append("Several repeated tokens detected.")
    if signals["latin_ratio"] > .45 and signals["devanagari_ratio"] < .25:
        flags.append("High proportion of Latin-script characters.")
    if signals["review_trigger_terms"]:
        flags.append("Generalizing/absolute language detected; human review suggested.")
    if not flags:
        flags.append("No configured surface-level review triggers detected.")

    if model is not None and not isinstance(model, Exception):
        raw = model.predict([normalized])[0]
        dimensions = {name: score_to_percent(raw[i]) for i, name in enumerate(DIMS)}
        model_type = "TF-IDF char n-grams + multi-output Logistic Regression"
        scoring_mode = "Trained model"
    else:
        labels = weak_labels(normalized)
        dimensions = {name: score_to_percent(labels[name]) for name in DIMS}
        model_type = "Transparent text-pattern heuristic baseline"
        scoring_mode = "Heuristic fallback (model artifact not bundled)"
        if isinstance(model, Exception):
            flags.append("Trained model could not be loaded; heuristic fallback is active.")

    overall = round(sum(dimensions.values()) / len(DIMS))
    return {
        "overall_score": overall,
        "dimensions": dimensions,
        "label": "Looks strong (baseline)" if overall >= 80 else (
            "Review recommended" if overall >= 60 else "Needs careful review"
        ),
        "flags": flags,
        "signals": signals,
        "model_type": model_type,
        "scoring_mode": scoring_mode,
        "disclaimer": (
            "Experimental weakly supervised estimates, not a definitive judgment of "
            "grammar, meaning, or culture. Semantic relevance needs the original prompt "
            "and cultural appropriateness requires human review."
        ),
    }


st.sidebar.markdown("### Dimensions")
st.sidebar.markdown("- Grammar\\n- Semantic relevance\\n- Fluency\\n- Cultural context")
st.sidebar.caption("Experimental baseline · human review recommended")
model = load_model()
if model is None:
    st.sidebar.info("Lightweight mode: trained model is not bundled.")
elif isinstance(model, Exception):
    st.sidebar.warning("Model loading failed; heuristic fallback is enabled.")
else:
    st.sidebar.success("Trained model loaded.")

st.markdown(
    '<div class="hero"><div class="eyebrow">HINDI NLP / QUALITY ASSURANCE</div>'
    '<h1>Hindi AI Content Quality</h1>'
    '<p class="muted">Four dimension-level estimates with transparent review signals.</p></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.15, .85], gap="large")
with left:
    st.subheader("Text to evaluate")
    text = st.text_area(
        "Hindi text",
        height=230,
        placeholder="यहाँ हिंदी टेक्स्ट लिखें या AI-generated content paste करें…",
        label_visibility="collapsed",
    )
    go = st.button("Evaluate content")

with right:
    st.subheader("What this checks")
    st.markdown(
        "**01 · Grammatical correctness**\\n\\nSurface language patterns."
        "\\n\\n**02 · Semantic relevance**\\n\\nA provisional text-pattern estimate; "
        "a source prompt is needed for true relevance."
        "\\n\\n**03 · Fluency**\\n\\nReadability and repetition signals."
        "\\n\\n**04 · Cultural appropriateness**\\n\\nHuman context-sensitive review is essential."
    )
    st.info("Scores are experimental; they are not objective truth.")

if go:
    if not text.strip():
        st.warning("Please enter text first.")
    else:
        try:
            with st.spinner("Evaluating text…"):
                out = evaluate_text(text, model)
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
        else:
            tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Dimensions", "Signals", "JSON"])
            with tab1:
                st.subheader("📊 Overall estimate")
                st.markdown(
                    f'<div class="metric"><div class="metric-label">Overall score</div>'
                    f'<div class="metric-value">{out["overall_score"]}'
                    f'<span style="font-size:.9rem"> / 100</span></div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'### {out["label"]}')
                st.caption(f'{out["scoring_mode"]} · Average of four dimension estimates.')
                st.caption(out["disclaimer"])
            with tab2:
                st.subheader("🔎 Dimension scores")
                columns = st.columns(4)
                labels = [
                    ("grammatical_correctness", "Grammar"),
                    ("semantic_relevance", "Relevance"),
                    ("fluency", "Fluency"),
                    ("cultural_appropriateness", "Cultural context"),
                ]
                for col, (key, label) in zip(columns, labels):
                    with col:
                        st.metric(label, f'{out["dimensions"][key]} / 100')
                        st.progress(out["dimensions"][key] / 100)
            with tab3:
                p, q = st.columns(2)
                with p:
                    st.subheader("⚠️ Review signals")
                    for flag in out["flags"]:
                        st.markdown("- " + flag)
                with q:
                    st.subheader("📑 Text signals")
                    st.json(out["signals"])
            with tab4:
                st.subheader("📂 Raw JSON")
                st.json(out)
                st.download_button(
                    "Download result JSON",
                    json.dumps(out, ensure_ascii=False, indent=2),
                    "hindi_quality_result.json",
                    "application/json",
                )
