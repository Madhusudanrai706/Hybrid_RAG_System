from ranking import (
    bm25_search,
    normalize_scores,
    weighted_hybrid_ranking
)


# ---------------------------------------------------
# Semantic Search
# ---------------------------------------------------

def semantic_search(vector_db, question, top_k=5):
    return vector_db.similarity_search(question, k=top_k)


# ---------------------------------------------------
# Keyword Search
# ---------------------------------------------------

def keyword_search(bm25, chunks, question, top_k=5):
    return bm25_search(bm25, chunks, question, top_k)


# ---------------------------------------------------
# Hybrid Retrieval
# ---------------------------------------------------

def hybrid_search(
    vector_db,
    bm25,
    chunks,
    question,
    semantic_weight=0.7,
    keyword_weight=0.3,
    top_k=5
):
    semantic_results = semantic_search(vector_db, question, top_k)
    keyword_results = keyword_search(bm25, chunks, question, top_k)

    return weighted_hybrid_ranking(
        semantic_results,
        keyword_results,
        semantic_weight,
        keyword_weight
    )


# ---------------------------------------------------
# Formatters — convert raw results into the same
# {"document", "semantic_score", "keyword_score", "final_score"}
# shape used everywhere in the UI, for the single-mode cases too.
# ---------------------------------------------------

def format_semantic_only(semantic_results):
    total = len(semantic_results)

    ranked = []
    for rank, doc in enumerate(semantic_results, start=1):
        score = (total - rank + 1) / total if total else 0
        ranked.append({
            "document": doc,
            "semantic_score": score,
            "keyword_score": 0,
            "final_score": score
        })
    return ranked


def format_keyword_only(keyword_results):
    if not keyword_results:
        return []

    scores = normalize_scores([score for score, _ in keyword_results])

    return [
        {
            "document": doc,
            "semantic_score": 0,
            "keyword_score": score,
            "final_score": score
        }
        for score, (_, doc) in zip(scores, keyword_results)
    ]
