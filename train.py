# train.py
from pathlib import Path
import json, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from src.quality import DIMS, weak_labels, normalize_text

ROOT = Path(__file__).resolve().parent

def main():
    data = ROOT / "data/hindi_corpus.csv"
    if not data.exists():
        raise SystemExit("Run python download_data.py --limit 5000 first")

    df = pd.read_csv(data).dropna(subset=["text"])
    df["text"] = df.text.map(normalize_text)
    df = df[df.text.str.len() >= 20].drop_duplicates("text")

    if len(df) < 50:
        raise SystemExit("Need at least 50 rows; download a larger sample.")

    # Generate weak labels (scores 1–5)
    y = pd.DataFrame([weak_labels(t) for t in df.text])

    # Ensure at least 2 unique values per dimension
    for d in DIMS:
        if y[d].nunique() < 2:
            raise SystemExit(f"Dimension {d} has only one class. Adjust weak_labels or dataset.")

    # Stratified split
    Xtr, Xte, ytr, yte = train_test_split(
        df.text, y[DIMS], test_size=0.2, stratify=y[DIMS], random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            max_features=100000,
            min_df=2,
            sublinear_tf=True
        )),
        ("classifier", MultiOutputClassifier(
            LogisticRegression(max_iter=300, class_weight="balanced")
        ))
    ])

    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    metrics = {}

    for i, d in enumerate(DIMS):
        metrics[d] = {
            "macro_f1": float(f1_score(yte.iloc[:, i], pred[:, i],
                                       average="macro", zero_division=0)),
            "report": classification_report(yte.iloc[:, i], pred[:, i],
                                            zero_division=0, output_dict=True)
        }
        print(d, round(metrics[d]["macro_f1"], 3))

    md = ROOT / "models"
    md.mkdir(exist_ok=True)
    joblib.dump(model, md / "quality_model.joblib")
    (md / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (md / "model_info.json").write_text(json.dumps({
        "model_type": "TF-IDF char n-grams + MultiOutput Logistic Regression",
        "training_rows": len(Xtr),
        "test_rows": len(Xte),
        "label_source": "weak heuristics (ordinal scores 1–5)"
    }, indent=2), encoding="utf-8")

    print("Model artifacts saved to models/")

if __name__ == "__main__":
    main()
