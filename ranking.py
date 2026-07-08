import numpy as np
from rank_bm25 import BM25Okapi


# --------------------------------------------------
# Build BM25 Index
# --------------------------------------------------

def build_bm25(chunks):

    documents = [
        chunk.page_content
        for chunk in chunks
    ]

    tokenized_docs = [
        doc.lower().split()
        for doc in documents
    ]

    bm25 = BM25Okapi(tokenized_docs)

    return bm25


# --------------------------------------------------
# BM25 Search
# --------------------------------------------------

def bm25_search(
    bm25,
    chunks,
    query,
    top_k=5
):

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True
    )

    return ranked[:top_k]


# --------------------------------------------------
# Normalize Scores
# --------------------------------------------------

def normalize_scores(scores):

    scores = np.array(scores)

    if len(scores) == 0:

        return scores

    minimum = scores.min()

    maximum = scores.max()

    if maximum == minimum:

        return np.ones_like(scores)

    return (scores - minimum) / (maximum - minimum)


# --------------------------------------------------
# Hybrid Ranking
# --------------------------------------------------

def weighted_hybrid_ranking(
    semantic_results,
    keyword_results,
    semantic_weight=0.7,
    keyword_weight=0.3
):

    hybrid_scores = {}

    # -----------------------
    # Semantic
    # -----------------------

    semantic_scores = [
        score
        for _, score in semantic_results
    ]

    semantic_scores = normalize_scores(
        semantic_scores
    )

    for i, (doc, _) in enumerate(semantic_results):

        hybrid_scores[doc.page_content] = {

            "document": doc,

            "semantic_score": semantic_scores[i],

            "keyword_score": 0,

            "final_score":
            semantic_weight * semantic_scores[i]

        }

    # -----------------------
    # Keyword
    # -----------------------

    keyword_scores = [
        score
        for score, _ in keyword_results
    ]

    keyword_scores = normalize_scores(
        keyword_scores
    )

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

                "final_score":
                keyword_weight * keyword_scores[i]

            }

    ranked = sorted(

        hybrid_scores.values(),

        key=lambda x: x["final_score"],

        reverse=True

    )

    return ranked