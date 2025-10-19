# utils.py -- tokenization, stopwords, simple helpers
import os
import re
from collections import defaultdict

# Try underthesea tokenizer
USE_UNDERTHESEA = False
try:
    from underthesea import word_tokenize as vn_word_tokenize
    USE_UNDERTHESEA = True
except Exception:
    USE_UNDERTHESEA = False

STOPWORDS_PATH = "vietnamese-stopwords.txt"

def load_stopwords(path=STOPWORDS_PATH):
    if not os.path.exists(path):
        return set(["và","là","của","có","cho","trong","với","đã","những","khi","một","về","tại","từ","vậy","như","vì"])
    with open(path, encoding="utf-8") as f:
        return set([line.strip() for line in f if line.strip()])

STOPWORDS = load_stopwords()

def simple_tokenize(text: str):
    if not isinstance(text, str):
        return []
    text = text.lower()
    # keep Vietnamese letters, numbers, spaces
    text = re.sub(r"[^a-zA-ZÀ-Ỵà-ỵ0-9\s]", " ", text)
    toks = text.split()
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 0]
    return toks

def tokenize(text: str):
    if not isinstance(text, str):
        return []
    if USE_UNDERTHESEA:
        try:
            s = vn_word_tokenize(text, format="text")
            toks = s.split()
            toks = [t for t in toks if t not in STOPWORDS and len(t) > 0]
            return toks
        except Exception:
            return simple_tokenize(text)
    else:
        return simple_tokenize(text)
