# src/quality.py
import re, unicodedata

# Dimensions we want to score
DIMS = [
    "grammatical_correctness",
    "semantic_relevance",
    "fluency",
    "cultural_appropriateness"
]

# Trigger terms that may reduce cultural appropriateness
TERMS = ["हमेशा", "कभी नहीं", "सभी लोग", "हर भारतीय", "असभ्य", "नीचा"]

def normalize_text(text: str) -> str:
    """Normalize Unicode and collapse whitespace."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text or ""))).strip()

def text_signals(text: str) -> dict:
    """Extract signals from text for weak labeling."""
    t = normalize_text(text)
    tokens = re.findall(r"[^\W_]+", t, re.UNICODE)
    chars = [c for c in t if not c.isspace()]

    dev = sum("\u0900" <= c <= "\u097f" for c in chars)
    lat = sum(c.lower() in "abcdefghijklmnopqrstuvwxyz" for c in chars)
    low = [x.lower() for x in tokens]

    return {
        "char_count": len(t),
        "token_count": len(tokens),
        "devanagari_ratio": dev / max(1, len(chars)),
        "latin_ratio": lat / max(1, len(chars)),
        "repeated_token_count": len(low) - len(set(low)),
        "repeated_punctuation": bool(re.search(r"([!?।,])\1{1,}", t)),
        "sentence_end_mark": t.endswith(("।", ".", "!", "?")),
        "review_trigger_terms": [x for x in TERMS if x in t],
    }

def weak_labels(text: str) -> dict:
    """Generate weak labels (scores 1–5) using heuristics."""
    t = normalize_text(text)
    s = text_signals(t)
    if not t:
        return {d: 1 for d in DIMS}

    g, f, r, c = 4, 4, 3, 4

    # Grammar / fluency rules
    if s["token_count"] < 4:
        g -= 1; f -= 1; r -= 1
    if s["repeated_punctuation"]:
        g -= 1; f -= 1
    if s["repeated_token_count"] >= 3:
        f -= 1
    if s["latin_ratio"] > .45 and s["devanagari_ratio"] < .25:
        g -= 1
    if not s["sentence_end_mark"] and s["token_count"] >= 7:
        g -= 1; f -= 1

    # Semantic relevance tweaks (ensure variation)
    if s["token_count"] >= 8:
        r += 1
    if s["token_count"] >= 15:
        r += 1   # extra boost for longer texts
    if any(term in t for term in TERMS):
        r -= 1   # penalize if trigger terms present
    if s["char_count"] > 300:
        r += 1   # very long text gets higher relevance

    # Cultural appropriateness
    if s["review_trigger_terms"]:
        c -= 1

    return dict(zip(DIMS, [max(1, min(5, x)) for x in [g, r, f, c]]))

def score_to_percent(x: float) -> int:
    """Convert score (1–5) to percentage (20–100)."""
    return int(round(max(1, min(5, float(x))) * 20))
