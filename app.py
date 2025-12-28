# app.py
import os
import json
from flask import Flask, render_template, request, jsonify
import pandas as pd

from indexer import build_and_cache_index, load_cached_index
from utils import tokenize, load_stopwords
import ranker, evaluator

DATA_PATH = "news_dataset.json"
GROUND_TRUTH_PATH = "ground_truth.json"

app = Flask(__name__)

# Load dataset (only metadata: title/content)
print("Loading dataset file:", DATA_PATH)
if not os.path.exists(DATA_PATH):
    print("Dataset not found. Put news_dataset.json in project folder.")
    df = pd.DataFrame(columns=["id","title","content","topic","source","url"])
else:
    df_raw = pd.read_json(DATA_PATH)
    # ensure required cols exist
    for c in ["id", "title", "content", "topic", "source", "url", "author"]:
        if c not in df_raw.columns:
            df_raw[c] = None
    df_raw = df_raw[["id", "title", "content", "topic", "source", "url", "author"]].dropna(subset=["content"]).reset_index(drop=True)
    df = df_raw

# Try load cached index; if not exist, build and cache
print("Loading or building index and tf-idf...")
data = load_cached_index()
if data is None:
    print("No cache found. Building index (this runs once)...")
    data = build_and_cache_index(df)
else:
    print("Cache loaded.")

# Helper: load ground truth
def load_ground_truth():
    if os.path.exists(GROUND_TRUTH_PATH):
        with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_ground_truth(gt):
    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search_route():
    query = request.form.get("query", "").strip()
    method = request.form.get("method", "cosine")
    k = int(request.form.get("k", 10))
    # tokenize query with same tokenizer
    q_tokens = tokenize(query)
    q_str = " ".join(q_tokens)
    
    # Mapping of ranking methods
    if method == "cosine":
        results = ranker.rank_cosine(q_str, data, top_k=k)
    elif method == "bm25":
        results = ranker.rank_bm25(q_tokens, data, top_k=k)
    elif method == "bm25_custom":
        results = ranker.bm25_custom(q_tokens, data, top_k=k)
    elif method == "bm25_plus":
        results = ranker.bm25_plus(q_tokens, data, top_k=k)
    elif method == "ql":
        results = ranker.query_likelihood_dirichlet(q_tokens, data, top_k=k)
    elif method == "ql_jm":
        results = ranker.query_likelihood_jelinek_mercer(q_tokens, data, top_k=k)
    elif method == "two_stage":
        results = ranker.two_stage_smoothing(q_tokens, data, top_k=k)
    elif method == "absolute_discount":
        results = ranker.absolute_discount(q_tokens, data, top_k=k)
    elif method == "odds":
        results = ranker.odds_likelihood_ratio(q_tokens, data, top_k=k)
    elif method == "pivoted":
        results = ranker.pivoted_length_normalization(q_tokens, data, top_k=k)
    elif method == "title_boost":
        results = ranker.title_boosted_ranking(q_tokens, data, top_k=k)
    elif method == "proximity":
        results = ranker.proximity_ranking(q_tokens, data, top_k=k)
    elif method == "dfr":
        results = ranker.divergence_from_randomness(q_tokens, data, top_k=k)
    elif method == "combined":
        results = ranker.combined_scoring(q_tokens, data, top_k=k)
    else:
        results = ranker.rank_cosine(q_str, data, top_k=k)

    enriched = []
    for r in results:
        docrow = data['df'][data['df']['id'].astype(str) == r['doc_id']].iloc[0]
        snippet = docrow['content'][:400].replace("\n"," ") + ("..." if len(docrow['content'])>400 else "")
        enriched.append({
            "doc_id": r['doc_id'],
            "title": docrow.get('title'),
            "score": r['score'],
            "topic": docrow.get('topic'),
            "source": docrow.get('source'),
            "author": docrow.get('author'),
            "url": docrow.get('url'),
            "snippet": snippet
        })
    return render_template("results.html", query=query, method=method, results=enriched, k=k)

@app.route("/mark", methods=["POST"])
def mark_relevance():
    payload = request.get_json()
    query = payload.get("query","").strip()
    doc_id = payload.get("doc_id")
    relevant = payload.get("relevant","true") == "true"
    if query == "" or doc_id is None:
        return jsonify({"ok": False}), 400
    gt = load_ground_truth()
    if query not in gt:
        gt[query] = []
    if relevant:
        if doc_id not in gt[query]:
            gt[query].append(doc_id)
    else:
        if doc_id in gt[query]:
            gt[query].remove(doc_id)
    save_ground_truth(gt)
    return jsonify({"ok": True})

@app.route("/eval", methods=["GET","POST"])
def evaluate():
    gt = load_ground_truth()
    if request.method == "GET":
        return render_template("eval.html", n_queries=len(gt), results=None)
    method = request.form.get("method","cosine")
    k = int(request.form.get("k",10))
    queries = list(gt.keys())
    actuals = []
    predicted_lists = []
    precisions = []
    recalls = []
    f1s = []
    p_at_k = []
    ndcgs = []
    for q in queries:
        rel_set = set(gt[q])
        actuals.append(rel_set)
        q_tokens = tokenize(q)
        q_str = " ".join(q_tokens)
        if method == "cosine":
            res = ranker.rank_cosine(q_str, data, top_k=len(data['df']))
        elif method == "bm25":
            res = ranker.rank_bm25(q_tokens, data, top_k=len(data['df']))
        elif method == "ql":
            res = ranker.query_likelihood_dirichlet(q_tokens, data, top_k=len(data['df']))
        elif method == "odds":
            res = ranker.odds_likelihood_ratio(q_tokens, data, top_k=len(data['df']))
        else:
            res = ranker.rank_cosine(q_str, data, top_k=len(data['df']))
        pred_list = [r['doc_id'] for r in res]
        predicted_lists.append(pred_list)
        p,r,f1 = evaluator.precision_recall_f1(rel_set, pred_list, k=k)
        precisions.append(p); recalls.append(r); f1s.append(f1)
        p_at_k.append(p); ndcgs.append(evaluator.ndcg_at_k(rel_set, pred_list, k))
    map_at_k = evaluator.mapk(actuals, predicted_lists, k)
    summary = {
        "method": method, "k": k, "n_queries": len(queries),
        "precision_mean": float(sum(precisions)/len(precisions)) if precisions else 0.0,
        "recall_mean": float(sum(recalls)/len(recalls)) if recalls else 0.0,
        "f1_mean": float(sum(f1s)/len(f1s)) if f1s else 0.0,
        "p_at_k_mean": float(sum(p_at_k)/len(p_at_k)) if p_at_k else 0.0,
        "map_at_k": float(map_at_k), "ndcg_mean": float(sum(ndcgs)/len(ndcgs)) if ndcgs else 0.0
    }
    return render_template("eval.html", n_queries=len(queries), results=summary)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
