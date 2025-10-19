# ranker.py
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# ranking functions expect data dict returned by indexer (with df, tfidf_vectorizer, tfidf_matrix, doc_term_counts, doc_lengths, collection_term_counts, total_terms_in_collection)

MU = 2000.0

def rank_cosine(query, data, top_k=10):
    q_proc = " ".join(query.split())  # assume query already tokenized string
    q_vec = data['tfidf_vectorizer'].transform([q_proc])
    sims = cosine_similarity(q_vec, data['tfidf_matrix']).flatten()
    top_idx = np.argsort(sims)[::-1][:top_k]
    res = []
    for idx in top_idx:
        row = data['df'].iloc[idx]
        res.append({"doc_id": str(row.get('id')), "title": row.get('title'), "score": float(sims[idx])})
    return res

def rank_bm25(query_tokens, data, top_k=10):
    tokenized_corpus = [ " ".join(toks).split() for toks in data['df']['tokens'].tolist() ]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    res = []
    for idx in top_idx:
        row = data['df'].iloc[idx]
        res.append({"doc_id": str(row.get('id')), "title": row.get('title'), "score": float(scores[idx])})
    return res

def query_likelihood_dirichlet(query_tokens, data, top_k=10):
    collection_prob = {}
    total_terms = data['total_terms_in_collection']
    for term, cnt in data['collection_term_counts'].items():
        collection_prob[term] = cnt / total_terms if total_terms>0 else 0.0

    scores = []
    for _, row in data['df'].iterrows():
        doc_id = str(row.get('id'))
        doc_len = data['doc_lengths'].get(doc_id, 0)
        doc_tc = data['doc_term_counts'].get(doc_id, {})
        denom = doc_len + MU
        logscore = 0.0
        for t in query_tokens:
            c_td = doc_tc.get(t, 0)
            p_c = collection_prob.get(t, 0.0)
            numer = c_td + MU * p_c
            logscore += math.log((numer/denom) + 1e-12)
        scores.append((doc_id, float(logscore)))
    scores.sort(key=lambda x: x[1], reverse=True)
    res = []
    for doc_id, s in scores[:top_k]:
        row = data['df'][data['df']['id'].astype(str) == doc_id].iloc[0]
        res.append({"doc_id": doc_id, "title": row.get('title'), "score": s})
    return res

def odds_likelihood_ratio(query_tokens, data, top_k=10):
    collection_prob = {}
    total_terms = data['total_terms_in_collection']
    for term, cnt in data['collection_term_counts'].items():
        collection_prob[term] = cnt / total_terms if total_terms>0 else 1e-12

    log_p_q_C = 0.0
    for t in query_tokens:
        log_p_q_C += math.log(collection_prob.get(t, 1e-12) + 1e-12)

    scores = []
    for _, row in data['df'].iterrows():
        doc_id = str(row.get('id'))
        doc_len = data['doc_lengths'].get(doc_id, 0)
        doc_tc = data['doc_term_counts'].get(doc_id, {})
        denom = doc_len + MU
        log_p_q_d = 0.0
        for t in query_tokens:
            c_td = doc_tc.get(t, 0)
            p_c = collection_prob.get(t, 1e-12)
            numer = c_td + MU * p_c
            log_p_q_d += math.log((numer/denom) + 1e-12)
        log_odds = log_p_q_d - log_p_q_C
        scores.append((doc_id, float(log_odds)))
    scores.sort(key=lambda x: x[1], reverse=True)
    res = []
    for doc_id, s in scores[:top_k]:
        row = data['df'][data['df']['id'].astype(str) == doc_id].iloc[0]
        res.append({"doc_id": doc_id, "title": row.get('title'), "score": s})
    return res
