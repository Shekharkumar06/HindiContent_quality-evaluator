import argparse
from pathlib import Path
import pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument("--limit",type=int,default=5000);p.add_argument("--output",default="data/hindi_corpus.csv");a=p.parse_args()
 from datasets import load_dataset
 print("Downloading open Hindi Wikipedia sample from wikimedia/wikipedia...")
 ds=load_dataset("wikimedia/wikipedia","20231101.hi",split=f"train[:{a.limit}]")
 rows=[{"text":str(x.get("text","")).strip()} for x in ds if len(str(x.get("text","")).strip())>=40]
 df=pd.DataFrame(rows).drop_duplicates("text");out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);df.to_csv(out,index=False,encoding="utf-8")
 print(f"Saved {len(df)} rows to {out.resolve()}. Labels are generated later using weak heuristics.")
if __name__=="__main__":main()
