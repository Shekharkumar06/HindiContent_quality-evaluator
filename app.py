import json, requests, streamlit as st

# --- Page setup ---
st.set_page_config(page_title="Hindi Content Quality", page_icon="◈", layout="wide")

# --- Custom CSS ---
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

# --- Sidebar ---
api = st.sidebar.text_input("API endpoint", "http://127.0.0.1:8000")
st.sidebar.markdown("**Dimensions**\n- Grammar\n- Semantic relevance\n- Fluency\n- Cultural context")
st.sidebar.caption("Experimental baseline · human review recommended")

# --- Hero section ---
st.markdown(
    '<div class="hero"><div class="eyebrow">HINDI NLP / QUALITY ASSURANCE</div>'
    '<h1>Hindi AI Content Quality</h1>'
    '<p class="muted">Four dimension-level estimates with transparent review signals.</p></div>',
    unsafe_allow_html=True
)

# --- Layout: input vs info ---
a, b = st.columns([1.15, .85], gap="large")
with a:
    st.subheader("Text to evaluate")
    text = st.text_area(
        "Hindi text",
        height=230,
        placeholder="यहाँ हिंदी टेक्स्ट लिखें या AI-generated content paste करें…",
        label_visibility="collapsed"
    )
    go = st.button("Evaluate content")

with b:
    st.subheader("What this checks")
    st.markdown("**01 · Grammatical correctness**\n\nSurface language patterns."
                "\n\n**02 · Semantic relevance**\n\nA provisional text-pattern estimate."
                "\n\n**03 · Fluency**\n\nReadability and repetition signals."
                "\n\n**04 · Cultural appropriateness**\n\nHuman context-sensitive review is essential.")
    st.info("Scores are experimental; they are not objective truth.")

# --- Evaluation ---
if go:
    if not text.strip():
        st.warning("Please enter text first.")
    else:
        with st.spinner("Evaluating..."):
            try:
                r = requests.post(api.rstrip("/") + "/evaluate", json={"text": text}, timeout=60)
            except requests.RequestException:
                st.error("API unavailable. Start: uvicorn backend.main:app --reload --port 8000")
            else:
                if r.status_code != 200:
                    st.error(f"API error {r.status_code}: {r.text}")
                else:
                    out = r.json()

                    # Tabs for results
                    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Dimensions", "Signals", "JSON"])

                    with tab1:
                        st.subheader("📊 Overall estimate")
                        st.markdown(
                            f'<div class="metric"><div class="metric-label">Overall score</div>'
                            f'<div class="metric-value">{out["overall_score"]}'
                            f'<span style="font-size:.9rem"> / 100</span></div></div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(f"### {out['label']}")
                        st.caption("Average of four model estimates.")

                    with tab2:
                        st.subheader("🔎 Dimension scores")
                        cols = st.columns(4)
                        labels = [
                            ("grammatical_correctness", "Grammar"),
                            ("semantic_relevance", "Relevance"),
                            ("fluency", "Fluency"),
                            ("cultural_appropriateness", "Cultural context")
                        ]
                        for c, (k, label) in zip(cols, labels):
                            with c:
                                st.metric(label, f"{out['dimensions'][k]} / 100")
                                st.progress(out["dimensions"][k] / 100)

                    with tab3:
                        p, q = st.columns(2)
                        with p:
                            st.subheader("⚠️ Review signals")
                            for f in out["flags"]:
                                st.markdown("- " + f)
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
                            "application/json"
                        )
                        st.caption(out["disclaimer"])
