# ranker.py
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from collections import Counter

# ranking functions expect data dict returned by indexer (with df, tfidf_vectorizer, tfidf_matrix, doc_term_counts, doc_lengths, collection_term_counts, total_terms_in_collection)

MU = 2000.0  # Dirichlet smoothing parameter
LAMBDA_JM = 0.7  # Jelinek-Mercer smoothing parameter
BM25_K1 = 1.5  # BM25 term frequency saturation
BM25_B = 0.75  # BM25 document length normalization

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


# ==================== NEW RANKING METHODS ====================

def query_likelihood_jelinek_mercer(query_tokens, data, top_k=10, lambda_jm=LAMBDA_JM):
    """
    Jelinek-Mercer Smoothing for Query Likelihood Model.
    P(w|d) = λ * P_ML(w|d) + (1-λ) * P(w|C)
    Good for short queries and balancing document vs collection probability.
    """
    collection_prob = {}
    total_terms = data['total_terms_in_collection']
    for term, cnt in data['collection_term_counts'].items():
        collection_prob[term] = cnt / total_terms if total_terms > 0 else 0.0

    scores = []
    for _, row in data['df'].iterrows():
        doc_id = str(row.get('id'))
        doc_len = data['doc_lengths'].get(doc_id, 0)
        doc_tc = data['doc_term_counts'].get(doc_id, {})
        logscore = 0.0
        for t in query_tokens:
            p_ml = doc_tc.get(t, 0) / doc_len if doc_len > 0 else 0.0
            p_c = collection_prob.get(t, 1e-12)
            p_smoothed = lambda_jm * p_ml + (1 - lambda_jm) * p_c
            logscore += math.log(p_smoothed + 1e-12)
        scores.append((doc_id, float(logscore)))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    res = []
    for doc_id, s in scores[:top_k]:
        row = data['df'][data['df']['id'].astype(str) == doc_id].iloc[0]
        res.append({"doc_id": doc_id, "title": row.get('title'), "score": s})
    return res


def bm25_custom(query_tokens, data, top_k=10, k1=BM25_K1, b=BM25_B):
    """
    Custom BM25 implementation with tunable parameters.
    Allows fine-tuning k1 (term saturation) and b (length normalization).
    """
    N = len(data['df'])
    avgdl = data['total_terms_in_collection'] / N if N > 0 else 1
    idf_dict = data.get('idf', {})
    
    scores = []
    for _, row in data['df'].iterrows():
        doc_id = str(row.get('id'))
        doc_len = data['doc_lengths'].get(doc_id, 0)
        doc_tc = data['doc_term_counts'].get(doc_id, {})
        score = 0.0
        for t in query_tokens:
            tf = doc_tc.get(t, 0)
            if tf == 0:
                continue
            idf = idf_dict.get(t, 0)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
            score += idf * (numerator / denominator)
        scores.append((doc_id, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    res = []
    for doc_id, s in scores[:top_k]:
        row = data['df'][data['df']['id'].astype(str) == doc_id].iloc[0]
        res.append({"doc_id": doc_id, "title": row.get('title'), "score": s})
    return res


def bm25_plus(query_tokens, data, top_k=10, k1=BM25_K1, b=BM25_B, delta=1.0):
    """
    BM25+ - Improved BM25 that addresses document length bias.
    Adds a lower bound (delta) to term frequency normalization.
    Better for long documents and prevents over-penalization.
    """
    N = len(data['df'])
    avgdl = data['total_terms_in_collection'] / N if N > 0 else 1
    idf_dict = data.get('idf', {})
    
    scores = []
    for _, row in data['df'].iterrows():
        doc_id = str(row.get('id'))
        doc_len = data['doc_lengths'].get(doc_id, 0)
        doc_tc = data['doc_term_counts'].get(doc_id, {})
        score = 0.0
        for t in query_tokens:
            tf = doc_tc.get(t, 0)
            if tf == 0:
                continue
            idf = idf_dict.get(t, 0)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
            # BM25+ adds delta to prevent negative tf normalization
            score += idf * ((numerator / denominator) + delta)
        scores.append((doc_id, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    res = []
    for doc_id, s in scores[:top_k]:
        row = data['df'][data['df']['id'].astype(str) == doc_id].iloc[0]
        res.append({"doc_id": doc_id, "title": row.get('title'), "score": s})
    return res


def pivoted_length_normalization(query_tokens, data, top_k=10, s=0.2):
    """
    Pivoted Document Length Normalization (Pivoted Unique).
    Uses pivot to adjust for document length bias.
    s controls the slope (0 = no normalization, 1 = full normalization).
    """
    N = len(data['df'])
    avgdl = data['total_terms_in_collection'] / N if N > 0 else 1
    idf_dict = data.get('idf', {})
    
    scores = []
    for _, row in data['df'].iterrows():
        doc_id = str(row.get('id'))
        doc_len = data['doc_lengths'].get(doc_id, 0)
        doc_tc = data['doc_term_counts'].get(doc_id, {})
        score = 0.0
        pivot_norm = (1 - s) + s * (doc_len / avgdl)
        for t in query_tokens:
            tf = doc_tc.get(t, 0)
            if tf == 0:
                continue
            idf = idf_dict.get(t, 0)
            tf_norm = (1 + math.log(1 + math.log(tf + 1))) / pivot_norm
            score += idf * tf_norm
        scores.append((doc_id, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    res = []
    for doc_id, s_val in scores[:top_k]:
        row = data['df'][data['df']['id'].astype(str) == doc_id].iloc[0]
        res.append({"doc_id": doc_id, "title": row.get('title'), "score": s_val})
    return res


def two_stage_smoothing(query_tokens, data, top_k=10, mu=MU, lambda_jm=LAMBDA_JM):
    """
    Two-Stage Smoothing Language Model.
    Combines Dirichlet (for estimation accuracy) with Jelinek-Mercer (for query modeling).
    Theoretically optimal for ad-hoc retrieval.
    """
    collection_prob = {}
    total_terms = data['total_terms_in_collection']
    for term, cnt in data['collection_term_counts'].items():
        collection_prob[term] = cnt / total_terms if total_terms > 0 else 0.0

    scores = []
    for _, row in data['df'].iterrows():
        doc_id = str(row.get('id'))
        doc_len = data['doc_lengths'].get(doc_id, 0)
        doc_tc = data['doc_term_counts'].get(doc_id, {})
        logscore = 0.0
        for t in query_tokens:
            c_td = doc_tc.get(t, 0)
            p_c = collection_prob.get(t, 1e-12)
            # Stage 1: Dirichlet smoothing
            p_dirichlet = (c_td + mu * p_c) / (doc_len + mu)
            # Stage 2: Jelinek-Mercer with Dirichlet as ML estimate
            p_two_stage = lambda_jm * p_dirichlet + (1 - lambda_jm) * p_c
            logscore += math.log(p_two_stage + 1e-12)
        scores.append((doc_id, float(logscore)))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    res = []
    for doc_id, s in scores[:top_k]:
        row = data['df'][data['df']['id'].astype(str) == doc_id].iloc[0]
        res.append({"doc_id": doc_id, "title": row.get('title'), "score": s})
    return res


def absolute_discount(query_tokens, data, top_k=10, delta=0.7):
    """
    Absolute Discount Smoothing.
    Subtracts a fixed discount (delta) from term counts and redistributes to unseen terms.
    Good for handling rare/unseen terms in queries.
    """
    collection_prob = {}
    total_terms = data['total_terms_in_collection']
    for term, cnt in data['collection_term_counts'].items():
        collection_prob[term] = cnt / total_terms if total_terms > 0 else 0.0

    scores = []
    for _, row in data['df'].iterrows():
        doc_id = str(row.get('id'))
        doc_len = data['doc_lengths'].get(doc_id, 0)
        doc_tc = data['doc_term_counts'].get(doc_id, {})
        unique_terms = len(doc_tc)
        logscore = 0.0
        for t in query_tokens:
            c_td = doc_tc.get(t, 0)
            p_c = collection_prob.get(t, 1e-12)
            if c_td > 0:
                p_term = max(c_td - delta, 0) / doc_len if doc_len > 0 else 0
            else:
                p_term = 0
            # Interpolation weight
            alpha = (delta * unique_terms) / doc_len if doc_len > 0 else 0
            p_smoothed = p_term + alpha * p_c
            logscore += math.log(p_smoothed + 1e-12)
        scores.append((doc_id, float(logscore)))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    res = []
    for doc_id, s in scores[:top_k]:
        row = data['df'][data['df']['id'].astype(str) == doc_id].iloc[0]
        res.append({"doc_id": doc_id, "title": row.get('title'), "score": s})
    return res


def title_boosted_ranking(query_tokens, data, top_k=10, title_weight=2.0):
    """
    Title-Boosted BM25 Ranking.
    Gives higher weight to terms matching in the title.
    Useful for newspaper articles where titles are highly informative.
    """
    from utils import tokenize
    
    N = len(data['df'])
    avgdl = data['total_terms_in_collection'] / N if N > 0 else 1
    idf_dict = data.get('idf', {})
    
    scores = []
    for _, row in data['df'].iterrows():
        doc_id = str(row.get('id'))
        doc_len = data['doc_lengths'].get(doc_id, 0)
        doc_tc = data['doc_term_counts'].get(doc_id, {})
        title_tokens = set(tokenize(str(row.get('title', ''))))
        
        score = 0.0
        for t in query_tokens:
            tf = doc_tc.get(t, 0)
            if tf == 0:
                continue
            idf = idf_dict.get(t, 0)
            numerator = tf * (BM25_K1 + 1)
            denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * (doc_len / avgdl))
            term_score = idf * (numerator / denominator)
            # Boost if term appears in title
            if t in title_tokens:
                term_score *= title_weight
            score += term_score
        scores.append((doc_id, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    res = []
    for doc_id, s in scores[:top_k]:
        row = data['df'][data['df']['id'].astype(str) == doc_id].iloc[0]
        res.append({"doc_id": doc_id, "title": row.get('title'), "score": s})
    return res


def proximity_ranking(query_tokens, data, top_k=10):
    """
    Proximity-Based Ranking.
    Rewards documents where query terms appear close together.
    Essential for phrase/multi-word queries.
    """
    inverted_index = data.get('inverted_index', {})
    idf_dict = data.get('idf', {})
    
    scores = []
    for _, row in data['df'].iterrows():
        doc_id = str(row.get('id'))
        doc_tc = data['doc_term_counts'].get(doc_id, {})
        
        # Base BM25 score
        base_score = 0.0
        for t in query_tokens:
            if doc_tc.get(t, 0) > 0:
                base_score += idf_dict.get(t, 0) * doc_tc.get(t, 0)
        
        # Proximity bonus
        positions = []
        for t in query_tokens:
            postings = inverted_index.get(t, [])
            for p in postings:
                if p['doc_id'] == doc_id:
                    positions.extend(p.get('positions', []))
                    break
        
        proximity_bonus = 0.0
        if len(positions) > 1:
            positions.sort()
            # Calculate minimum span containing all query terms
            min_span = positions[-1] - positions[0] + 1
            if min_span > 0:
                proximity_bonus = len(query_tokens) / min_span
        
        scores.append((doc_id, base_score + proximity_bonus * 10))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    res = []
    for doc_id, s in scores[:top_k]:
        row = data['df'][data['df']['id'].astype(str) == doc_id].iloc[0]
        res.append({"doc_id": doc_id, "title": row.get('title'), "score": s})
    return res


def divergence_from_randomness(query_tokens, data, top_k=10):
    """
    Divergence From Randomness (DFR) Model - PL2 variant.
    Based on probability that term distribution deviates from random.
    Theoretically grounded alternative to BM25.
    """
    N = len(data['df'])
    total_terms = data['total_terms_in_collection']
    avgdl = total_terms / N if N > 0 else 1
    
    scores = []
    for _, row in data['df'].iterrows():
        doc_id = str(row.get('id'))
        doc_len = data['doc_lengths'].get(doc_id, 0)
        doc_tc = data['doc_term_counts'].get(doc_id, {})
        score = 0.0
        
        c = 1.0  # normalization parameter
        for t in query_tokens:
            tf = doc_tc.get(t, 0)
            if tf == 0:
                continue
            
            cf = data['collection_term_counts'].get(t, 0)
            lambda_t = cf / N if N > 0 else 0
            
            # Poisson model for DFR
            tfn = tf * math.log(1 + c * avgdl / doc_len) if doc_len > 0 else tf
            
            if lambda_t > 0 and tfn > 0:
                # Information content
                info = -math.log(lambda_t + 1e-12)
                # Term frequency weight
                tf_weight = tfn / (tfn + 1)
                score += info * tf_weight
        
        scores.append((doc_id, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    res = []
    for doc_id, s in scores[:top_k]:
        row = data['df'][data['df']['id'].astype(str) == doc_id].iloc[0]
        res.append({"doc_id": doc_id, "title": row.get('title'), "score": s})
    return res


def combined_scoring(query_tokens, data, top_k=10, weights=None):
    """
    Ensemble/Combined Scoring.
    Combines multiple ranking methods with configurable weights.
    Often outperforms single methods.
    """
    if weights is None:
        weights = {'bm25': 0.4, 'ql': 0.3, 'cosine': 0.3}
    
    # Get scores from different methods
    q_str = " ".join(query_tokens)
    bm25_results = rank_bm25(query_tokens, data, top_k=len(data['df']))
    ql_results = query_likelihood_dirichlet(query_tokens, data, top_k=len(data['df']))
    cosine_results = rank_cosine(q_str, data, top_k=len(data['df']))
    
    # Normalize and combine
    all_docs = set()
    bm25_scores = {}
    ql_scores = {}
    cosine_scores = {}
    
    for r in bm25_results:
        bm25_scores[r['doc_id']] = r['score']
        all_docs.add(r['doc_id'])
    for r in ql_results:
        ql_scores[r['doc_id']] = r['score']
        all_docs.add(r['doc_id'])
    for r in cosine_results:
        cosine_scores[r['doc_id']] = r['score']
        all_docs.add(r['doc_id'])
    
    # Min-max normalization
    def normalize(scores_dict):
        if not scores_dict:
            return {}
        vals = list(scores_dict.values())
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return {k: 1.0 for k in scores_dict}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores_dict.items()}
    
    bm25_norm = normalize(bm25_scores)
    ql_norm = normalize(ql_scores)
    cosine_norm = normalize(cosine_scores)
    
    combined = []
    for doc_id in all_docs:
        score = (weights['bm25'] * bm25_norm.get(doc_id, 0) +
                 weights['ql'] * ql_norm.get(doc_id, 0) +
                 weights['cosine'] * cosine_norm.get(doc_id, 0))
        combined.append((doc_id, score))
    
    combined.sort(key=lambda x: x[1], reverse=True)
    res = []
    for doc_id, s in combined[:top_k]:
        row = data['df'][data['df']['id'].astype(str) == doc_id].iloc[0]
        res.append({"doc_id": doc_id, "title": row.get('title'), "score": s})
    return res
