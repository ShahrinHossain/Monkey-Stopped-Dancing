# backend/clir/evaluation.py
"""
Module D — Ranking, Scoring, & Evaluation (Core)

What this file does (as required by the assignment screenshots):

1) Ranking & Scoring
- Implements a ranking function that outputs a sorted list of top-K documents for each query.
- Outputs a Matching Confidence score (0–1) for each document.
- Normalizes model scores to [0, 1] before combining.
- Shows a low-confidence warning if the top result confidence < threshold (default 0.20).

2) Evaluation Metrics (Mandatory)
- Precision@10
- Recall@50
- nDCG@10
- MRR

How to run:
  cd backend
  python -m clir.evaluation --help

Typical run (with a qrels file):
  python -m clir.evaluation --qrels data/eval/qrels.jsonl --model hybrid --top_k 10

QRELS (ground-truth) file format:
- JSONL (one JSON per line), each line like:
  {
    "query": "good and bad news of sylhet",
    "relevant_urls": [
      "https://example.com/news1",
      "https://example.com/news2"
    ]
  }

Why URLs?
- Your dataset records contain "url" and it is stable for matching across languages/models.

Notes:
- This module does NOT do web search (Google/Bing) because that requires external API/web access.
  For your report, you can manually compare a few queries with Google/Bing and write the analysis.

Dependencies:
- Uses your Module C engine:
    from clir.query_retrieval import QueryRetrievalEngine
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Iterable, Set
import argparse
import json
import os
import math

import numpy as np

from clir.query_retrieval import QueryRetrievalEngine


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class RankedDocument:
    doc_id: int
    language: str
    title: str
    url: str
    date: Optional[str]
    model: str

    # normalized (0..1) score for the model chosen
    matching_confidence: float

    # optional debug fields copied from Module C results
    raw_score: float
    matched_keywords: List[str]
    evidence_lines: List[str]


@dataclass
class QueryEvaluationResult:
    query: str
    ranked_documents: List[RankedDocument]
    warning_low_confidence: bool
    top_confidence: float

    precision_at_10: Optional[float] = None
    recall_at_50: Optional[float] = None
    ndcg_at_10: Optional[float] = None
    mrr: Optional[float] = None


# -----------------------------
# Utilities
# -----------------------------

def minmax_normalize_scores(raw_scores: List[float]) -> List[float]:
    if not raw_scores:
        return []
    min_value = float(min(raw_scores))
    max_value = float(max(raw_scores))
    if math.isclose(min_value, max_value):
        # all same => confidence becomes 0 for all (no discriminative power)
        return [0.0 for _ in raw_scores]
    return [(float(score) - min_value) / (max_value - min_value) for score in raw_scores]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def flatten_results_by_model(module_c_output: Dict[str, Any], model_name: str) -> List[Dict[str, Any]]:
    """
    Takes Module C engine output and extracts the result list for:
      - output["en"][model_name]
      - output["bn"][model_name]
    Then merges them into one list.
    """
    combined: List[Dict[str, Any]] = []
    for language_key in ("en", "bn"):
        language_block = module_c_output.get(language_key, {})
        if isinstance(language_block, dict) and model_name in language_block:
            model_results = language_block.get(model_name, [])
            if isinstance(model_results, list):
                combined.extend(model_results)
    return combined


def deduplicate_by_url_keep_best(documents: List[RankedDocument]) -> List[RankedDocument]:
    """
    If same URL appears multiple times (e.g., from multiple languages/models),
    keep the highest-confidence version.
    """
    best_by_url: Dict[str, RankedDocument] = {}
    for doc in documents:
        if not doc.url:
            continue
        existing = best_by_url.get(doc.url)
        if existing is None or doc.matching_confidence > existing.matching_confidence:
            best_by_url[doc.url] = doc
    # sort again
    return sorted(best_by_url.values(), key=lambda d: d.matching_confidence, reverse=True)


# -----------------------------
# Ranking & scoring (Module D core)
# -----------------------------

class RankingAndScoringEngine:
    """
    Produces a single ranked list with confidence scores in [0,1].

    You can choose one model ("bm25"/"tfidf"/"fuzzy"/"semantic"/"hybrid")
    OR evaluate multiple models by calling repeatedly.
    """

    def __init__(
        self,
        retrieval_engine: Optional[QueryRetrievalEngine] = None,
        low_confidence_threshold: float = 0.20,
    ) -> None:
        self.retrieval_engine = retrieval_engine or QueryRetrievalEngine()
        self.low_confidence_threshold = low_confidence_threshold

    def rank(
        self,
        user_query: str,
        model_name: str = "hybrid",
        top_k: int = 10,
        include_debug: bool = True,
        hybrid_weights: Optional[Dict[str, float]] = None,
    ) -> QueryEvaluationResult:
        """
        Returns:
          - ranked list of top_k docs (merged across bn/en)
          - normalized confidence score for each doc
          - low-confidence warning if needed
        """
        module_c_output = self.retrieval_engine.search(
            user_query=user_query,
            top_k=max(top_k, 50),  # get enough docs for recall@50
            model=model_name if model_name != "all" else "all",
            include_debug=include_debug,
            hybrid_weights=hybrid_weights,
        )

        # If model_name == "all", we still need one model to rank.
        # Default to "hybrid" in that case.
        selected_model = model_name if model_name != "all" else "hybrid"

        raw_results = flatten_results_by_model(module_c_output, selected_model)

        raw_scores = [safe_float(item.get("score", 0.0)) for item in raw_results]
        normalized_confidences = minmax_normalize_scores(raw_scores)

        ranked_documents: List[RankedDocument] = []
        for item, confidence in zip(raw_results, normalized_confidences):
            ranked_documents.append(
                RankedDocument(
                    doc_id=int(item.get("doc_id", -1)),
                    language=str(item.get("language", "")),
                    title=str(item.get("title", "")),
                    url=str(item.get("url", "")),
                    date=item.get("date", None),
                    model=str(item.get("model", selected_model)),
                    matching_confidence=float(confidence),
                    raw_score=safe_float(item.get("score", 0.0)),
                    matched_keywords=list(item.get("matched_keywords", []) or []),
                    evidence_lines=list(item.get("evidence_lines", []) or []),
                )
            )

        # sort by confidence
        ranked_documents.sort(key=lambda d: d.matching_confidence, reverse=True)

        # deduplicate by URL (keep best)
        ranked_documents = deduplicate_by_url_keep_best(ranked_documents)

        # trim
        ranked_documents = ranked_documents[:top_k]

        top_confidence = ranked_documents[0].matching_confidence if ranked_documents else 0.0
        warning_low_confidence = bool(ranked_documents) and (top_confidence < self.low_confidence_threshold)

        return QueryEvaluationResult(
            query=user_query,
            ranked_documents=ranked_documents,
            warning_low_confidence=warning_low_confidence,
            top_confidence=top_confidence,
        )


# -----------------------------
# Evaluation metrics (Mandatory)
# -----------------------------

def precision_at_k(ranked_urls: List[str], relevant_urls: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = ranked_urls[:k]
    if not top_k:
        return 0.0
    relevant_in_top_k = sum(1 for url in top_k if url in relevant_urls)
    return relevant_in_top_k / float(k)


def recall_at_k(ranked_urls: List[str], relevant_urls: Set[str], k: int) -> float:
    if not relevant_urls:
        return 0.0
    top_k = ranked_urls[:k]
    retrieved_relevant = sum(1 for url in top_k if url in relevant_urls)
    return retrieved_relevant / float(len(relevant_urls))


def dcg_at_k(binary_relevance: List[int], k: int) -> float:
    """
    DCG with log2 discount:
      DCG = sum_{i=1..k} rel_i / log2(i+1)
    """
    dcg_value = 0.0
    for rank_index in range(min(k, len(binary_relevance))):
        rel = float(binary_relevance[rank_index])
        dcg_value += rel / math.log2(rank_index + 2)
    return dcg_value


def ndcg_at_k(ranked_urls: List[str], relevant_urls: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevance_list = [1 if url in relevant_urls else 0 for url in ranked_urls[:k]]
    ideal_list = sorted(relevance_list, reverse=True)

    dcg_value = dcg_at_k(relevance_list, k)
    idcg_value = dcg_at_k(ideal_list, k)

    if math.isclose(idcg_value, 0.0):
        return 0.0
    return dcg_value / idcg_value


def mean_reciprocal_rank(ranked_urls: List[str], relevant_urls: Set[str]) -> float:
    """
    MRR for a single query:
      1 / rank_of_first_relevant
    """
    for index, url in enumerate(ranked_urls):
        if url in relevant_urls:
            return 1.0 / float(index + 1)
    return 0.0


# -----------------------------
# QRELS Loading
# -----------------------------

def load_qrels_jsonl(qrels_path: str) -> List[Dict[str, Any]]:
    """
    Each line:
      {"query": "...", "relevant_urls": ["...", "..."]}
    """
    items: List[Dict[str, Any]] = []
    if not os.path.exists(qrels_path):
        raise FileNotFoundError(f"QRELS file not found: {qrels_path}")

    with open(qrels_path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "query" in obj and "relevant_urls" in obj:
                items.append(obj)
    return items


# -----------------------------
# Evaluator
# -----------------------------

class Evaluator:
    def __init__(
        self,
        model_name: str = "hybrid",
        top_k_for_ranking: int = 10,
        low_confidence_threshold: float = 0.20,
    ) -> None:
        self.model_name = model_name
        self.top_k_for_ranking = top_k_for_ranking
        self.ranking_engine = RankingAndScoringEngine(
            retrieval_engine=QueryRetrievalEngine(),
            low_confidence_threshold=low_confidence_threshold,
        )

    def evaluate_queries(self, qrels_items: List[Dict[str, Any]]) -> Tuple[List[QueryEvaluationResult], Dict[str, float]]:
        all_results: List[QueryEvaluationResult] = []

        precision_scores: List[float] = []
        recall_scores: List[float] = []
        ndcg_scores: List[float] = []
        mrr_scores: List[float] = []

        for item in qrels_items:
            query_text = str(item.get("query", "")).strip()
            relevant_urls_list = item.get("relevant_urls", []) or []
            relevant_urls = set(str(u).strip() for u in relevant_urls_list if str(u).strip())

            ranking_result = self.ranking_engine.rank(
                user_query=query_text,
                model_name=self.model_name,
                top_k=max(self.top_k_for_ranking, 50),  # for Recall@50
                include_debug=False,
            )

            ranked_urls = [doc.url for doc in ranking_result.ranked_documents if doc.url]

            # Metrics (mandatory)
            p10 = precision_at_k(ranked_urls, relevant_urls, k=10)
            r50 = recall_at_k(ranked_urls, relevant_urls, k=50)
            ndcg10 = ndcg_at_k(ranked_urls, relevant_urls, k=10)
            mrr_value = mean_reciprocal_rank(ranked_urls, relevant_urls)

            ranking_result.precision_at_10 = p10
            ranking_result.recall_at_50 = r50
            ranking_result.ndcg_at_10 = ndcg10
            ranking_result.mrr = mrr_value

            all_results.append(ranking_result)

            precision_scores.append(p10)
            recall_scores.append(r50)
            ndcg_scores.append(ndcg10)
            mrr_scores.append(mrr_value)

        summary = {
            "queries_count": float(len(all_results)),
            "Precision@10": float(np.mean(precision_scores)) if precision_scores else 0.0,
            "Recall@50": float(np.mean(recall_scores)) if recall_scores else 0.0,
            "nDCG@10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
            "MRR": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        }
        return all_results, summary


# -----------------------------
# Pretty printing (for your demo / report screenshots)
# -----------------------------

def print_ranked_results(query_result: QueryEvaluationResult) -> None:
    print("\n" + "=" * 90)
    print(f"Query: {query_result.query}")

    if query_result.warning_low_confidence:
        print(
            f"⚠ Warning: Retrieved results may not be relevant. Matching confidence is low "
            f"(score: {query_result.top_confidence:.2f})."
        )
        print("Consider rephrasing your query or checking translation quality.")

    print("-" * 90)
    for rank_index, doc in enumerate(query_result.ranked_documents, start=1):
        print(f"{rank_index}. confidence={doc.matching_confidence:.4f} | raw={doc.raw_score:.4f} | {doc.title}")
        print(f"    url: {doc.url}")
        if doc.matched_keywords:
            print(f"    matched keywords: {', '.join(doc.matched_keywords)}")
        for line in (doc.evidence_lines or [])[:3]:
            print(f"    line: {line}")

    if query_result.precision_at_10 is not None:
        print("-" * 90)
        print(
            f"Precision@10={query_result.precision_at_10:.3f} | "
            f"Recall@50={query_result.recall_at_50:.3f} | "
            f"nDCG@10={query_result.ndcg_at_10:.3f} | "
            f"MRR={query_result.mrr:.3f}"
        )


def print_summary(summary: Dict[str, float]) -> None:
    print("\n" + "#" * 90)
    print("EVALUATION SUMMARY")
    print("#" * 90)
    print(f"Queries: {int(summary.get('queries_count', 0))}")
    print(f"Precision@10: {summary.get('Precision@10', 0.0):.3f}")
    print(f"Recall@50:    {summary.get('Recall@50', 0.0):.3f}")
    print(f"nDCG@10:      {summary.get('nDCG@10', 0.0):.3f}")
    print(f"MRR:          {summary.get('MRR', 0.0):.3f}")


# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Module D — Ranking, Scoring & Evaluation")
    parser.add_argument(
        "--qrels",
        type=str,
        default="",
        help="Path to qrels JSONL. Each line: {query:'...', relevant_urls:[...]}",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hybrid",
        choices=["bm25", "tfidf", "fuzzy", "semantic", "hybrid"],
        help="Which retrieval model to rank/evaluate.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-K documents to show per query (ranking output).",
    )
    parser.add_argument(
        "--low_conf",
        type=float,
        default=0.20,
        help="Low-confidence threshold for warning (default 0.20).",
    )
    parser.add_argument(
        "--demo_query",
        type=str,
        default="",
        help="Run a single query demo without qrels (prints ranked results + warning).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    evaluator = Evaluator(
        model_name=args.model,
        top_k_for_ranking=args.top_k,
        low_confidence_threshold=args.low_conf,
    )

    # Demo mode (no qrels needed)
    if args.demo_query.strip():
        demo_result = evaluator.ranking_engine.rank(
            user_query=args.demo_query.strip(),
            model_name=args.model,
            top_k=args.top_k,
            include_debug=False,
        )
        print_ranked_results(demo_result)
        return

    # Evaluation mode (qrels required)
    if not args.qrels.strip():
        raise SystemExit(
            "You must provide --qrels for evaluation, or use --demo_query for demo.\n"
            "Example:\n"
            "  python -m clir.evaluation --demo_query \"good and bad news of sylhet\" --model hybrid\n"
            "  python -m clir.evaluation --qrels data/eval/qrels.jsonl --model hybrid"
        )

    qrels_items = load_qrels_jsonl(args.qrels.strip())
    all_results, summary = evaluator.evaluate_queries(qrels_items)

    for query_result in all_results:
        # show ranking + per-query metrics
        print_ranked_results(query_result)

    # show summary metrics
    print_summary(summary)


if __name__ == "__main__":
    main()
