# Sample output

Illustrative response shape only; scores vary by trained model.

```json
{
  "overall_score": 80,
  "dimensions": {
    "grammatical_correctness": 80,
    "semantic_relevance": 60,
    "fluency": 80,
    "cultural_appropriateness": 80
  },
  "label": "Looks strong (baseline)",
  "flags": ["No configured surface-level review triggers detected."]
}
```

Scores map provisional 1–5 classes to percentages. Semantic relevance requires the original prompt. No flags does not prove content is culturally appropriate.
