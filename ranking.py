import numpy as np


# --------------------------------------------------
# BM25 Search
# (the BM25 index itself is built in embeddings.get_sparse_embeddings)
# --------------------------------------------------

def bm25_search(bm25, chunks, query, top_k=5):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True
    )

    return ranked[:top_k]


# --------------------------------------------------
# Normalize scores to a 0-1 range
# --------------------------------------------------

def normalize_scores(scores):
    scores = np.array(scores, dtype=float)

    if len(scores) == 0:
        return scores

    minimum, maximum = scores.min(), scores.max()

    if maximum == minimum:
        return np.ones_like(scores)

    return (scores - minimum) / (maximum - minimum)


# --------------------------------------------------
# Hybrid Ranking (weighted fusion of semantic + keyword)
# --------------------------------------------------

def weighted_hybrid_ranking(
    semantic_results,
    keyword_results,
    semantic_weight=0.7,
    keyword_weight=0.3
):
    hybrid_scores = {}

    # Semantic results (rank-based score, since FAISS doesn't return
    # a normalized similarity score by default via similarity_search)
    semantic_scores = normalize_scores(
        [1 / (i + 1) for i in range(len(semantic_results))]
    )

    for i, doc in enumerate(semantic_results):
        hybrid_scores[doc.page_content] = {
            "document": doc,
            "semantic_score": semantic_scores[i],
            "keyword_score": 0,
            "final_score": semantic_weight * semantic_scores[i]
        }

    # BM25 results
    keyword_scores = normalize_scores([score for score, _ in keyword_results])

    for i, (score, doc) in enumerate(keyword_results):
        if doc.page_content in hybrid_scores:
            hybrid_scores[doc.page_content]["keyword_score"] = keyword_scores[i]
            hybrid_scores[doc.page_content]["final_score"] += (
                keyword_weight * keyword_scores[i]
            )
        else:
            hybrid_scores[doc.page_content] = {
                "document": doc,
                "semantic_score": 0,
                "keyword_score": keyword_scores[i],
                "final_score": keyword_weight * keyword_scores[i]
            }

    return sorted(
        hybrid_scores.values(),
        key=lambda x: x["final_score"],
        reverse=True
    )
