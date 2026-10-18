from pathlib import Path
import json,joblib
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from src.quality import DIMS,normalize_text,text_signals,score_to_percent
ROOT=Path(__file__).resolve().parents[1];MP=ROOT/"models/quality_model.joblib";app=FastAPI(title="Hindi AI Content Quality API");model=None
class Payload(BaseModel): text:str=Field(...,min_length=1,max_length=10000)
@app.get("/health")
def health():return {"status":"ok","model_loaded":MP.exists()}
@app.get("/model-info")
def info():
 p=ROOT/"models/model_info.json"
 return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"model_type":"not trained","message":"Run download_data.py and train.py."}
@app.post("/evaluate")
def evaluate(p:Payload):
 global model
 t=normalize_text(p.text)
 if not t:raise HTTPException(422,"Text cannot be empty")
 if model is None:
  if not MP.exists():raise HTTPException(503,"Model not found. Download data and train first.")
  model=joblib.load(MP)
 raw=model.predict([t])[0];dims={d:score_to_percent(raw[i]) for i,d in enumerate(DIMS)};s=text_signals(t);flags=[]
 if s["token_count"]<4:flags.append("Text is very short; estimates may be unreliable.")
 if s["repeated_punctuation"]:flags.append("Repeated punctuation detected.")
 if s["repeated_token_count"]>=3:flags.append("Several repeated tokens detected.")
 if s["latin_ratio"]>.45 and s["devanagari_ratio"]<.25:flags.append("High proportion of Latin-script characters.")
 if s["review_trigger_terms"]:flags.append("Generalizing/absolute language detected; human review suggested.")
 if not flags:flags.append("No configured surface-level review triggers detected.")
 score=round(sum(dims.values())/4)
 return {"overall_score":score,"dimensions":dims,"label":"Looks strong (baseline)" if score>=80 else ("Review recommended" if score>=60 else "Needs careful review"),"flags":flags,"signals":s,"model_type":"TF-IDF char n-grams + multi-output Logistic Regression","disclaimer":"Weakly supervised estimates, not a definitive judgment of grammar, meaning, or culture."}
