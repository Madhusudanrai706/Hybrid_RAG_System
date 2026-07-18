import numpy as np


# --------------------------------------------------
# Metadata Filtering (Feature 2)
# --------------------------------------------------
# Applied BEFORE ranking/slicing to top_k, so a relevant chunk that
# happens to score low overall (but is the only match within the
# filtered scope) isn't discarded before the filter even gets a chance
# to look at it.

def filter_documents(chunks, sources=None, page_range=None, chapter=None, subject=None):
    """
    Keep only chunks matching ALL given (non-None) criteria.
      - sources:     iterable of filenames to restrict to
      - page_range:  (min_page, max_page) inclusive; chunks with page=None
                     are excluded if this filter is active
      - chapter:     exact chapter string match (case-insensitive)
      - subject:     exact subject string match (case-insensitive)
    Any filter left as None/empty means "no constraint on this field".
    """
    if not any([sources, page_range, chapter, subject]):
        return list(chunks)

    filtered = []

    for chunk in chunks:
        meta = chunk.metadata

        if sources and meta.get("source") not in sources:
            continue

        if page_range:
            page = meta.get("page")
            if page is None or not (page_range[0] <= page <= page_range[1]):
                continue

        if chapter and meta.get("chapter", "Unknown").lower() != chapter.lower():
            continue

        if subject and meta.get("subject", "General").lower() != subject.lower():
            continue

        filtered.append(chunk)

    return filtered


# --------------------------------------------------
# BM25 Search
# (the BM25 index itself is built in embeddings.get_sparse_embeddings)
# --------------------------------------------------

def bm25_search(bm25, chunks, query, top_k=5):
    """Original unfiltered BM25 search - kept for backward compatibility."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True
    )

    return ranked[:top_k]


def bm25_search_filtered(bm25, chunks, query, top_k=5, sources=None,
                          page_range=None, chapter=None, subject=None):
    """
    BM25 search with metadata filtering applied before the top_k cut.

    BM25 scores are computed over the FULL corpus (the index was built
    that way), so we can't pre-filter the chunk list before scoring - the
    score array is positionally aligned with the original corpus. Instead:
    score everything, pair scores with chunks, filter the pairs by
    metadata, THEN sort and slice to top_k.
    """
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    pairs = list(zip(scores, chunks))

    if any([sources, page_range, chapter, subject]):
        allowed = {
            id(c) for c in filter_documents(
                chunks, sources=sources, page_range=page_range,
                chapter=chapter, subject=subject
            )
        }
        pairs = [(score, chunk) for score, chunk in pairs if id(chunk) in allowed]

    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs[:top_k]


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