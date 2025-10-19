# indexer.py (phiên bản tối ưu & có log chi tiết)
import os
import pickle
import math
import time
from collections import defaultdict, Counter
from tqdm import tqdm
import pandas as pd
from scipy.sparse import save_npz, load_npz
from utils import tokenize

CACHE_INDEX = "cache_index.pkl"
CACHE_TFIDF_VEC = "tfidf_vectorizer.pkl"
CACHE_TFIDF_MATRIX = "tfidf_matrix.npz"
TOKENIZED_JSON = "news_tokenized.json"


def build_and_cache_index(df, force_rebuild=False):
    """
    Xây dựng inverted index, thống kê TF/IDF và cache vào file để lần sau dùng lại.
    """
    if (
        not force_rebuild
        and os.path.exists(CACHE_INDEX)
        and os.path.exists(CACHE_TFIDF_VEC)
        and os.path.exists(CACHE_TFIDF_MATRIX)
        and os.path.exists(TOKENIZED_JSON)
    ):
        print("⚙️  Loading cached index and TF-IDF...")
        with open(CACHE_INDEX, "rb") as f:
            index_data = pickle.load(f)
        with open(CACHE_TFIDF_VEC, "rb") as f:
            tfidf_vectorizer = pickle.load(f)
        tfidf_matrix = load_npz(CACHE_TFIDF_MATRIX)
        index_data.update(
            {"tfidf_vectorizer": tfidf_vectorizer, "tfidf_matrix": tfidf_matrix}
        )
        return index_data

    # --- TOKENIZE ---
    print("🔹 Tokenizing documents (this may take a while)...")
    if os.path.exists(TOKENIZED_JSON) and not force_rebuild:
        df = pd.read_json(TOKENIZED_JSON)
        tokens_col = df["tokens"].tolist()
    else:
        tokens_col = []
        for content in tqdm(df["content"].tolist(), total=len(df)):
            toks = tokenize(content)
            tokens_col.append(toks)
        df2 = df.copy()
        df2["tokens"] = tokens_col
        df2.to_json(TOKENIZED_JSON, orient="records", force_ascii=False)
        df = df2

    # --- BUILD INDEX ---
    print("🔹 Building doc term counts and collection stats...")
    inverted_index = defaultdict(list)
    doc_term_counts = {}
    doc_lengths = {}
    collection_term_counts = Counter()
    total_terms_in_collection = 0

    for i, row in tqdm(df.iterrows(), total=len(df)):
        doc_id = str(row.get("id", i))
        tokens = row["tokens"]
        doc_lengths[doc_id] = len(tokens)
        tc = Counter(tokens)
        doc_term_counts[doc_id] = tc
        collection_term_counts.update(tc)
        total_terms_in_collection += len(tokens)

    print("🔹 Building inverted index postings (with positions)...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        doc_id = str(row.get("id", i))
        tokens = row["tokens"]
        pos_map = defaultdict(list)
        for pos, t in enumerate(tokens):
            pos_map[t].append(pos)
        for term, pos_list in pos_map.items():
            posting = {"doc_id": doc_id, "tf": len(pos_list), "positions": pos_list}
            inverted_index[term].append(posting)

    print("🔹 Computing IDF for terms...")
    N = len(df)
    idf = {}
    for term, postings in inverted_index.items():
        df_t = len(postings)
        idf_val = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)
        idf[term] = idf_val

    # --- TF-IDF ---
    print("🔹 Fitting TF-IDF vectorizer (this may take a while but only once)...")
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus_for_vectorizer = [" ".join(toks) for toks in tokens_col]
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_for_vectorizer)

    # --- SAVE CACHE ---
    print("💾 Saving caches to disk...")
    start_time = time.time()
    try:
        print("  → Writing main index and stats...")
        with open(CACHE_INDEX, "wb") as f:
            pickle.dump(
                {
                    "inverted_index": inverted_index,
                    "doc_term_counts": doc_term_counts,
                    "doc_lengths": doc_lengths,
                    "collection_term_counts": collection_term_counts,
                    "total_terms_in_collection": total_terms_in_collection,
                    "idf": idf,
                    "df": df[["id", "title", "content", "topic", "source", "url", "author", "tokens"]],
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print("  → Writing TF-IDF vectorizer...")
        with open(CACHE_TFIDF_VEC, "wb") as f:
            pickle.dump(tfidf_vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("  → Writing TF-IDF matrix...")
        save_npz(CACHE_TFIDF_MATRIX, tfidf_matrix)

        elapsed = time.time() - start_time
        print(f"✅ Cache saved successfully in {elapsed:.2f}s")

    except KeyboardInterrupt:
        print("\n⚠️ Saving interrupted by user! Partial cache may be corrupted.")
        return None
    except Exception as e:
        print(f"\n❌ Error while saving cache: {e}")
        return None

    return {
        "inverted_index": inverted_index,
        "doc_term_counts": doc_term_counts,
        "doc_lengths": doc_lengths,
        "collection_term_counts": collection_term_counts,
        "total_terms_in_collection": total_terms_in_collection,
        "idf": idf,
        "tfidf_vectorizer": tfidf_vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "df": df,
    }


def load_cached_index():
    """Đọc cache nếu có, trả về None nếu lỗi hoặc file bị hỏng."""
    if not os.path.exists(CACHE_INDEX):
        return None
    try:
        with open(CACHE_INDEX, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"⚠️ Cache file corrupted or incomplete: {e}")
        return None

    try:
        with open(CACHE_TFIDF_VEC, "rb") as f:
            tfidf_vectorizer = pickle.load(f)
        tfidf_matrix = load_npz(CACHE_TFIDF_MATRIX)
        data.update({"tfidf_vectorizer": tfidf_vectorizer, "tfidf_matrix": tfidf_matrix})
        print("✅ Cache loaded successfully.")
        return data
    except Exception as e:
        print(f"⚠️ Error loading TF-IDF components: {e}")
        return None
