from src.quality import normalize_text,text_signals,weak_labels,DIMS
def test_normalize(): assert normalize_text("  नमस्ते   दुनिया  ")=="नमस्ते दुनिया"
def test_labels(): 
 x=weak_labels("यह एक साधारण हिंदी वाक्य है।")
 assert set(x)==set(DIMS) and all(1<=v<=5 for v in x.values())
def test_punctuation(): assert text_signals("क्या!!!")["repeated_punctuation"]
