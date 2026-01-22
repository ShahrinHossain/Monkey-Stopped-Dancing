# # 
# # backend/clir/query_retrieval.py
# """
# Module C — Retrieval Models (Core)

# This file:
# - Calls Module B QueryProcessor to get:
#     - retrieval_queries  ✅ (clean queries for retrieval)
#     - retrieval_keywords ✅ (important keywords only; stopwords removed)
# - Loads your dataset from:
#     backend/data/processed/en.jsonl
#     backend/data/processed/bn.jsonl
# - Implements and compares retrieval models:
#   Model 1: Lexical Retrieval (TF-IDF + BM25)
#   Model 2: Fuzzy/Transliteration Matching (RapidFuzz or difflib fallback)
#   Model 3: Semantic Matching (sentence-transformers embeddings)
#   Model 4: Hybrid Ranking (weighted fusion)

# ✅ FIX INCLUDED:
# - Evidence matching now uses ONLY `retrieval_keywords` from Module B.
# - Words like: what/where/are/the/and will NOT be used for matching (because Module B removed them).
# - Also uses word-boundary matching for English (so partial matches don't trigger).

# Run:
#   cd backend
#   python -m clir.query_retrieval
# """

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Dict, List, Optional, Tuple, Any
# import os
# import json
# import math
# import time
# import re

# import numpy as np

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import minmax_scale

# from clir.query_processor import QueryProcessor, QueryProcessingResult


# # -----------------------------
# # Paths / Config
# # -----------------------------

# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# BACKEND_DIR = os.path.dirname(CURRENT_DIR)

# DEFAULT_BN_JSONL_PATH = os.path.join(BACKEND_DIR, "data", "processed", "bn.jsonl")
# DEFAULT_EN_JSONL_PATH = os.path.join(BACKEND_DIR, "data", "processed", "en.jsonl")

# DEFAULT_EMBEDDING_CACHE_DIR = os.path.join(BACKEND_DIR, "models", "embeddings_cache")
# os.makedirs(DEFAULT_EMBEDDING_CACHE_DIR, exist_ok=True)

# _WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)


# def _normalize_text_for_indexing(text: str) -> str:
#     text = (text or "").strip()
#     text = _WHITESPACE_RE.sub(" ", text)
#     return text


# # -----------------------------
# # Data Structures
# # -----------------------------

# @dataclass
# class DocumentRecord:
#     doc_id: int
#     language: str  # 'bn' or 'en'
#     title: str
#     body: str
#     url: str
#     date: Optional[str] = None

#     @property
#     def full_text(self) -> str:
#         title = _normalize_text_for_indexing(self.title)
#         body = _normalize_text_for_indexing(self.body)
#         return f"{title}. {title}. {body}".strip()


# @dataclass
# class ScoredResult:
#     doc_id: int
#     language: str
#     url: str
#     title: str
#     date: Optional[str]
#     score: float
#     model: str

#     # ✅ Evidence fields for debugging/demo
#     matched_keywords: List[str]
#     evidence_lines: List[str]


# # -----------------------------
# # JSONL Loader
# # -----------------------------

# def load_jsonl_documents(jsonl_path: str, language_code: str, starting_doc_id: int = 0) -> List[DocumentRecord]:
#     documents: List[DocumentRecord] = []
#     doc_id_counter = starting_doc_id

#     if not os.path.exists(jsonl_path):
#         raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

#     with open(jsonl_path, "r", encoding="utf-8") as file_handle:
#         for line in file_handle:
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 record = json.loads(line)
#             except json.JSONDecodeError:
#                 continue

#             title = str(record.get("title", "") or "")
#             body = str(record.get("body", "") or "")
#             url = str(record.get("url", "") or "")
#             date = record.get("date", None)

#             if not (title or body):
#                 continue

#             documents.append(
#                 DocumentRecord(
#                     doc_id=doc_id_counter,
#                     language=language_code,
#                     title=title,
#                     body=body,
#                     url=url,
#                     date=date,
#                 )
#             )
#             doc_id_counter += 1

#     return documents


# # -----------------------------
# # Model 1: TF-IDF
# # -----------------------------

# class TfidfRetriever:
#     def __init__(self, documents: List[DocumentRecord]) -> None:
#         self.documents = documents
#         self.vectorizer = TfidfVectorizer(
#             lowercase=False,
#             min_df=2,
#             max_df=0.95,
#             ngram_range=(1, 2),
#         )
#         self.document_matrix = None  # scipy sparse

#     def build(self) -> None:
#         corpus = [doc.full_text for doc in self.documents]
#         self.document_matrix = self.vectorizer.fit_transform(corpus)

#     def search(self, query_text: str, top_k: int = 10) -> Tuple[List[int], np.ndarray]:
#         if self.document_matrix is None:
#             raise RuntimeError("TF-IDF retriever not built. Call build() first.")

#         query_vector = self.vectorizer.transform([query_text])
#         scores = (self.document_matrix @ query_vector.T).toarray().reshape(-1)

#         if top_k <= 0:
#             top_k = 10

#         top_indices = np.argsort(-scores)[:top_k]
#         return top_indices.tolist(), scores


# # -----------------------------
# # Model 1: BM25 (Okapi)
# # -----------------------------

# class BM25Retriever:
#     def __init__(self, documents: List[DocumentRecord], k1: float = 1.5, b: float = 0.75) -> None:
#         self.documents = documents
#         self.k1 = k1
#         self.b = b

#         self.tokenized_documents: List[List[str]] = []
#         self.term_document_frequency: Dict[str, int] = {}
#         self.document_term_frequencies: List[Dict[str, int]] = []
#         self.document_lengths: List[int] = []
#         self.average_document_length: float = 0.0

#     @staticmethod
#     def _tokenize(text: str) -> List[str]:
#         text = _normalize_text_for_indexing(text)
#         return [t for t in text.split(" ") if t]

#     def build(self) -> None:
#         self.tokenized_documents = []
#         self.term_document_frequency = {}
#         self.document_term_frequencies = []
#         self.document_lengths = []

#         for doc in self.documents:
#             tokens = self._tokenize(doc.full_text)
#             self.tokenized_documents.append(tokens)
#             self.document_lengths.append(len(tokens))

#             term_counts: Dict[str, int] = {}
#             unique_terms_in_doc = set()

#             for term in tokens:
#                 term_counts[term] = term_counts.get(term, 0) + 1
#                 unique_terms_in_doc.add(term)

#             self.document_term_frequencies.append(term_counts)

#             for term in unique_terms_in_doc:
#                 self.term_document_frequency[term] = self.term_document_frequency.get(term, 0) + 1

#         total_length = sum(self.document_lengths) if self.document_lengths else 0
#         self.average_document_length = (total_length / len(self.documents)) if self.documents else 0.0

#     def _idf(self, term: str) -> float:
#         document_count = len(self.documents)
#         doc_freq = self.term_document_frequency.get(term, 0)
#         return math.log(1 + (document_count - doc_freq + 0.5) / (doc_freq + 0.5))

#     def search(self, query_text: str, top_k: int = 10) -> Tuple[List[int], np.ndarray]:
#         query_tokens = self._tokenize(query_text)
#         scores = np.zeros(len(self.documents), dtype=np.float32)

#         for doc_index, term_counts in enumerate(self.document_term_frequencies):
#             doc_length = self.document_lengths[doc_index] if self.document_lengths else 0
#             length_norm = (1 - self.b) + self.b * (doc_length / (self.average_document_length + 1e-9))

#             score = 0.0
#             for term in query_tokens:
#                 tf = term_counts.get(term, 0)
#                 if tf == 0:
#                     continue

#                 idf = self._idf(term)
#                 numerator = tf * (self.k1 + 1)
#                 denominator = tf + self.k1 * length_norm
#                 score += idf * (numerator / (denominator + 1e-9))

#             scores[doc_index] = score

#         top_indices = np.argsort(-scores)[:top_k]
#         return top_indices.tolist(), scores


# # -----------------------------
# # Model 2: Fuzzy / Transliteration Matching
# # -----------------------------

# class FuzzyRetriever:
#     def __init__(self, documents: List[DocumentRecord]) -> None:
#         self.documents = documents

#         self._use_rapidfuzz = False
#         self._rapidfuzz_fuzz = None

#         try:
#             from rapidfuzz import fuzz as rapidfuzz_fuzz  # type: ignore
#             self._use_rapidfuzz = True
#             self._rapidfuzz_fuzz = rapidfuzz_fuzz
#         except Exception:
#             self._use_rapidfuzz = False

#         self.document_search_strings: List[str] = []
#         for doc in self.documents:
#             preview_body = _normalize_text_for_indexing(doc.body)[:400]
#             combined = f"{doc.title} {preview_body} {doc.url}"
#             self.document_search_strings.append(combined)

#     def search(self, query_text: str, top_k: int = 10) -> Tuple[List[int], np.ndarray]:
#         query_text = _normalize_text_for_indexing(query_text)
#         scores = np.zeros(len(self.documents), dtype=np.float32)

#         if self._use_rapidfuzz and self._rapidfuzz_fuzz is not None:
#             scorer = self._rapidfuzz_fuzz.token_sort_ratio
#             for doc_index, search_text in enumerate(self.document_search_strings):
#                 similarity = scorer(query_text, search_text)  # 0..100
#                 scores[doc_index] = float(similarity) / 100.0
#         else:
#             import difflib
#             for doc_index, search_text in enumerate(self.document_search_strings):
#                 similarity = difflib.SequenceMatcher(None, query_text, search_text).ratio()
#                 scores[doc_index] = float(similarity)

#         top_indices = np.argsort(-scores)[:top_k]
#         return top_indices.tolist(), scores


# # -----------------------------
# # Model 3: Semantic Matching (sentence-transformers)
# # -----------------------------

# class SemanticRetriever:
#     def __init__(
#         self,
#         documents: List[DocumentRecord],
#         embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
#         cache_dir: str = DEFAULT_EMBEDDING_CACHE_DIR,
#         cache_key: str = "default",
#     ) -> None:
#         self.documents = documents
#         self.embedding_model_name = embedding_model_name
#         self.cache_dir = cache_dir
#         self.cache_key = cache_key

#         self._model = None
#         self.document_embeddings: Optional[np.ndarray] = None

#         os.makedirs(self.cache_dir, exist_ok=True)

#     def _load_model(self):
#         if self._model is not None:
#             return self._model
#         from sentence_transformers import SentenceTransformer  # type: ignore
#         self._model = SentenceTransformer(self.embedding_model_name)
#         return self._model

#     def _cache_path(self) -> str:
#         safe_key = re.sub(r"[^a-zA-Z0-9_\-]+", "_", self.cache_key)
#         safe_model = re.sub(r"[^a-zA-Z0-9_\-]+", "_", self.embedding_model_name)
#         return os.path.join(self.cache_dir, f"emb_{safe_model}_{safe_key}.npz")

#     def build(self, force_rebuild: bool = False) -> None:
#         cache_path = self._cache_path()

#         if (not force_rebuild) and os.path.exists(cache_path):
#             cached = np.load(cache_path)
#             self.document_embeddings = cached["embeddings"]
#             return

#         embedding_model = self._load_model()
#         document_texts = [doc.full_text for doc in self.documents]
#         embeddings = embedding_model.encode(document_texts, convert_to_numpy=True, show_progress_bar=True)
#         embeddings = embeddings.astype(np.float32)

#         norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
#         embeddings = embeddings / norms

#         self.document_embeddings = embeddings
#         np.savez_compressed(cache_path, embeddings=embeddings)

#     def search(self, query_text: str, top_k: int = 10) -> Tuple[List[int], np.ndarray]:
#         if self.document_embeddings is None:
#             raise RuntimeError("Semantic retriever not built. Call build() first.")

#         embedding_model = self._load_model()
#         query_embedding = embedding_model.encode([query_text], convert_to_numpy=True).astype(np.float32)
#         query_embedding = query_embedding / (np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-9)

#         scores = (self.document_embeddings @ query_embedding.T).reshape(-1)
#         top_indices = np.argsort(-scores)[:top_k]
#         return top_indices.tolist(), scores


# # -----------------------------
# # Hybrid Fusion
# # -----------------------------

# def _normalize_scores(scores: np.ndarray) -> np.ndarray:
#     if scores.size == 0:
#         return scores
#     if np.all(scores == scores[0]):
#         return np.zeros_like(scores, dtype=np.float32)
#     return minmax_scale(scores.astype(np.float32))


# def fuse_scores(
#     bm25_scores: np.ndarray,
#     tfidf_scores: np.ndarray,
#     fuzzy_scores: np.ndarray,
#     semantic_scores: np.ndarray,
#     weights: Dict[str, float],
# ) -> np.ndarray:
#     bm25_norm = _normalize_scores(bm25_scores)
#     tfidf_norm = _normalize_scores(tfidf_scores)
#     fuzzy_norm = _normalize_scores(fuzzy_scores)
#     semantic_norm = _normalize_scores(semantic_scores)

#     final_scores = (
#         weights.get("bm25", 0.0) * bm25_norm
#         + weights.get("tfidf", 0.0) * tfidf_norm
#         + weights.get("fuzzy", 0.0) * fuzzy_norm
#         + weights.get("semantic", 0.0) * semantic_norm
#     )
#     return final_scores.astype(np.float32)


# # -----------------------------
# # Evidence Matching (FIXED)
# # -----------------------------

# _SENTENCE_SPLIT_RE = re.compile(r"(?:\r?\n)+|(?<=[\.\!\?।])\s+", flags=re.UNICODE)

# def _build_english_keyword_regex(keyword: str) -> re.Pattern:
#     escaped = re.escape(keyword.lower())
#     # word boundary for English-like tokens
#     return re.compile(rf"\b{escaped}\b", flags=re.IGNORECASE)

# def find_evidence_lines_for_document(
#     document: DocumentRecord,
#     language_code: str,
#     keywords: List[str],
#     max_lines: int = 3,
# ) -> Tuple[List[str], List[str]]:
#     """
#     Returns (matched_keywords, evidence_lines)
#     ✅ Uses ONLY keywords provided (which come from Module B retrieval_keywords).
#     ✅ For English uses word boundary matching.
#     ✅ For Bangla uses substring match (works reasonably for BN script).
#     """
#     keywords = [k.strip() for k in (keywords or []) if k and k.strip()]
#     if not keywords:
#         return [], []

#     title_text = (document.title or "")
#     body_text = (document.body or "")
#     combined_text = f"{title_text}\n{body_text}"

#     matched_keywords: List[str] = []

#     if language_code == "en":
#         keyword_patterns = [(kw, _build_english_keyword_regex(kw)) for kw in keywords]
#         lower_title = title_text.lower()
#         lower_body = body_text.lower()

#         for kw, pattern in keyword_patterns:
#             if pattern.search(lower_title) or pattern.search(lower_body):
#                 matched_keywords.append(kw)
#     else:
#         # BN/mixed: simple substring
#         for kw in keywords:
#             if kw in title_text or kw in body_text:
#                 matched_keywords.append(kw)

#     if not matched_keywords:
#         return [], []

#     # Split into "sentences/lines" for evidence
#     raw_parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(combined_text) if p and p.strip()]
#     evidence_lines: List[str] = []

#     if language_code == "en":
#         patterns = [_build_english_keyword_regex(kw) for kw in matched_keywords]
#         for part in raw_parts:
#             part_lower = part.lower()
#             if any(p.search(part_lower) for p in patterns):
#                 evidence_lines.append(part)
#             if len(evidence_lines) >= max_lines:
#                 break
#     else:
#         for part in raw_parts:
#             if any(kw in part for kw in matched_keywords):
#                 evidence_lines.append(part)
#             if len(evidence_lines) >= max_lines:
#                 break

#     return matched_keywords, evidence_lines


# # -----------------------------
# # Query Retrieval Engine (Module C)
# # -----------------------------

# class QueryRetrievalEngine:
#     """
#     Models supported:
#       - "bm25"
#       - "tfidf"
#       - "fuzzy"
#       - "semantic"
#       - "hybrid"
#       - "all"
#     """

#     def __init__(
#         self,
#         bangla_jsonl_path: str = DEFAULT_BN_JSONL_PATH,
#         english_jsonl_path: str = DEFAULT_EN_JSONL_PATH,
#         embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
#         enable_stopwords: bool = True,
#     ) -> None:
#         self.bangla_jsonl_path = bangla_jsonl_path
#         self.english_jsonl_path = english_jsonl_path
#         self.embedding_model_name = embedding_model_name

#         self.query_processor = QueryProcessor(enable_stopwords=enable_stopwords)

#         self.documents_bn: List[DocumentRecord] = []
#         self.documents_en: List[DocumentRecord] = []

#         self.tfidf_bn: Optional[TfidfRetriever] = None
#         self.tfidf_en: Optional[TfidfRetriever] = None

#         self.bm25_bn: Optional[BM25Retriever] = None
#         self.bm25_en: Optional[BM25Retriever] = None

#         self.fuzzy_bn: Optional[FuzzyRetriever] = None
#         self.fuzzy_en: Optional[FuzzyRetriever] = None

#         self.semantic_bn: Optional[SemanticRetriever] = None
#         self.semantic_en: Optional[SemanticRetriever] = None

#         self._load_documents()
#         self._build_retrievers()

#     def _load_documents(self) -> None:
#         self.documents_bn = load_jsonl_documents(self.bangla_jsonl_path, "bn", starting_doc_id=0)
#         self.documents_en = load_jsonl_documents(self.english_jsonl_path, "en", starting_doc_id=len(self.documents_bn))

#     def _build_retrievers(self) -> None:
#         self.tfidf_bn = TfidfRetriever(self.documents_bn)
#         self.tfidf_bn.build()

#         self.tfidf_en = TfidfRetriever(self.documents_en)
#         self.tfidf_en.build()

#         self.bm25_bn = BM25Retriever(self.documents_bn)
#         self.bm25_bn.build()

#         self.bm25_en = BM25Retriever(self.documents_en)
#         self.bm25_en.build()

#         self.fuzzy_bn = FuzzyRetriever(self.documents_bn)
#         self.fuzzy_en = FuzzyRetriever(self.documents_en)

#         self.semantic_bn = SemanticRetriever(
#             self.documents_bn,
#             embedding_model_name=self.embedding_model_name,
#             cache_key="bn",
#         )
#         self.semantic_bn.build(force_rebuild=False)

#         self.semantic_en = SemanticRetriever(
#             self.documents_en,
#             embedding_model_name=self.embedding_model_name,
#             cache_key="en",
#         )
#         self.semantic_en.build(force_rebuild=False)

#     def _format_results(
#         self,
#         documents: List[DocumentRecord],
#         scores: np.ndarray,
#         top_indices: List[int],
#         model_name: str,
#         language_code: str,
#         retrieval_keywords: List[str],
#     ) -> List[ScoredResult]:
#         results: List[ScoredResult] = []
#         for doc_index in top_indices:
#             doc = documents[doc_index]

#             matched_keywords, evidence_lines = find_evidence_lines_for_document(
#                 document=doc,
#                 language_code=language_code,
#                 keywords=retrieval_keywords,
#                 max_lines=3,
#             )

#             results.append(
#                 ScoredResult(
#                     doc_id=doc.doc_id,
#                     language=language_code,
#                     url=doc.url,
#                     title=doc.title,
#                     date=doc.date,
#                     score=float(scores[doc_index]),
#                     model=model_name,
#                     matched_keywords=matched_keywords,
#                     evidence_lines=evidence_lines,
#                 )
#             )
#         return results

#     def _search_single_language(
#         self,
#         language_code: str,
#         retrieval_queries: List[str],
#         retrieval_keywords: List[str],
#         top_k: int,
#         model: str,
#         hybrid_weights: Optional[Dict[str, float]] = None,
#     ) -> Dict[str, List[ScoredResult]]:
#         if language_code == "bn":
#             documents = self.documents_bn
#             tfidf_retriever = self.tfidf_bn
#             bm25_retriever = self.bm25_bn
#             fuzzy_retriever = self.fuzzy_bn
#             semantic_retriever = self.semantic_bn
#         else:
#             documents = self.documents_en
#             tfidf_retriever = self.tfidf_en
#             bm25_retriever = self.bm25_en
#             fuzzy_retriever = self.fuzzy_en
#             semantic_retriever = self.semantic_en

#         if tfidf_retriever is None or bm25_retriever is None or fuzzy_retriever is None or semantic_retriever is None:
#             raise RuntimeError("Retrievers not initialized properly.")

#         def max_over_variants(search_fn):
#             best_scores = np.zeros(len(documents), dtype=np.float32)
#             for query_text in retrieval_queries:
#                 _, scores = search_fn(query_text, top_k=len(documents))
#                 best_scores = np.maximum(best_scores, scores.astype(np.float32))
#             return best_scores

#         outputs: Dict[str, List[ScoredResult]] = {}

#         tfidf_scores = max_over_variants(tfidf_retriever.search) if model in ("tfidf", "all", "hybrid") else np.zeros(len(documents), dtype=np.float32)
#         bm25_scores = max_over_variants(bm25_retriever.search) if model in ("bm25", "all", "hybrid") else np.zeros(len(documents), dtype=np.float32)
#         fuzzy_scores = max_over_variants(fuzzy_retriever.search) if model in ("fuzzy", "all", "hybrid") else np.zeros(len(documents), dtype=np.float32)
#         semantic_scores = max_over_variants(semantic_retriever.search) if model in ("semantic", "all", "hybrid") else np.zeros(len(documents), dtype=np.float32)

#         if model in ("tfidf", "all"):
#             top_indices = np.argsort(-tfidf_scores)[:top_k].tolist()
#             outputs["tfidf"] = self._format_results(documents, tfidf_scores, top_indices, "tfidf", language_code, retrieval_keywords)

#         if model in ("bm25", "all"):
#             top_indices = np.argsort(-bm25_scores)[:top_k].tolist()
#             outputs["bm25"] = self._format_results(documents, bm25_scores, top_indices, "bm25", language_code, retrieval_keywords)

#         if model in ("fuzzy", "all"):
#             top_indices = np.argsort(-fuzzy_scores)[:top_k].tolist()
#             outputs["fuzzy"] = self._format_results(documents, fuzzy_scores, top_indices, "fuzzy", language_code, retrieval_keywords)

#         if model in ("semantic", "all"):
#             top_indices = np.argsort(-semantic_scores)[:top_k].tolist()
#             outputs["semantic"] = self._format_results(documents, semantic_scores, top_indices, "semantic", language_code, retrieval_keywords)

#         if model in ("hybrid", "all"):
#             if hybrid_weights is None:
#                 hybrid_weights = {"bm25": 0.30, "tfidf": 0.10, "fuzzy": 0.20, "semantic": 0.40}

#             hybrid_scores = fuse_scores(
#                 bm25_scores=bm25_scores,
#                 tfidf_scores=tfidf_scores,
#                 fuzzy_scores=fuzzy_scores,
#                 semantic_scores=semantic_scores,
#                 weights=hybrid_weights,
#             )
#             top_indices = np.argsort(-hybrid_scores)[:top_k].tolist()
#             outputs["hybrid"] = self._format_results(documents, hybrid_scores, top_indices, "hybrid", language_code, retrieval_keywords)

#         return outputs

#     def search(
#         self,
#         user_query: str,
#         top_k: int = 10,
#         model: str = "all",
#         hybrid_weights: Optional[Dict[str, float]] = None,
#         include_debug: bool = True,
#     ) -> Dict[str, Any]:
#         if model not in ("bm25", "tfidf", "fuzzy", "semantic", "hybrid", "all"):
#             raise ValueError("model must be one of: bm25|tfidf|fuzzy|semantic|hybrid|all")

#         start_time = time.perf_counter()

#         module_b_result: QueryProcessingResult = self.query_processor.process(user_query)
#         module_b_dict = module_b_result.to_dict()

#         # ✅ FIX: Use ONLY retrieval_queries from Module B for retrieval
#         retrieval_queries_bn = module_b_dict.get("retrieval_queries", {}).get("bn", []) or []
#         retrieval_queries_en = module_b_dict.get("retrieval_queries", {}).get("en", []) or []

#         # ✅ FIX: Use ONLY retrieval_keywords from Module B for evidence matching
#         retrieval_keywords_bn = module_b_dict.get("retrieval_keywords", {}).get("bn", []) or []
#         retrieval_keywords_en = module_b_dict.get("retrieval_keywords", {}).get("en", []) or []

#         # Fallback safety (should rarely happen)
#         if not retrieval_queries_bn:
#             retrieval_queries_bn = module_b_result.expanded_queries.get("bn", [])[:1] or [module_b_result.normalized_query]
#         if not retrieval_queries_en:
#             retrieval_queries_en = module_b_result.expanded_queries.get("en", [])[:1] or [module_b_result.normalized_query]

#         module_b_time = time.perf_counter()

#         bn_results = self._search_single_language(
#             language_code="bn",
#             retrieval_queries=retrieval_queries_bn[:5],
#             retrieval_keywords=retrieval_keywords_bn,
#             top_k=top_k,
#             model=model,
#             hybrid_weights=hybrid_weights,
#         )
#         bn_time = time.perf_counter()

#         en_results = self._search_single_language(
#             language_code="en",
#             retrieval_queries=retrieval_queries_en[:5],
#             retrieval_keywords=retrieval_keywords_en,
#             top_k=top_k,
#             model=model,
#             hybrid_weights=hybrid_weights,
#         )
#         en_time = time.perf_counter()

#         timings_ms = {
#             "module_b_processing": (module_b_time - start_time) * 1000.0,
#             "bn_retrieval": (bn_time - module_b_time) * 1000.0,
#             "en_retrieval": (en_time - bn_time) * 1000.0,
#             "total": (en_time - start_time) * 1000.0,
#         }

#         response: Dict[str, Any] = {
#             "bn": {k: [r.__dict__ for r in v] for k, v in bn_results.items()},
#             "en": {k: [r.__dict__ for r in v] for k, v in en_results.items()},
#             "timings_ms": timings_ms,
#         }
#         if include_debug:
#             response["module_b"] = module_b_dict

#         return response


# # -----------------------------
# # Minimal CLI test
# # -----------------------------

# def _print_results_block(language_label: str, model_name: str, results: List[Dict[str, Any]]) -> None:
#     print(f"\n--- {language_label} | {model_name.upper()} ---")
#     for rank, item in enumerate(results, start=1):
#         print(f"{rank}. score={item['score']:.4f} | {item['title']}")
#         print(f"    url: {item['url']}")
#         matched = item.get("matched_keywords", []) or []
#         if matched:
#             print(f"    matched keywords: {', '.join(matched)}")
#         evidence = item.get("evidence_lines", []) or []
#         for line in evidence:
#             print(f"    line: {line}")


# if __name__ == "__main__":
#     retrieval_engine = QueryRetrievalEngine()

#     while True:
#         user_query = input("Query (empty to quit): ").strip()
#         if not user_query:
#             break

#         output = retrieval_engine.search(user_query, top_k=5, model="all", include_debug=True)

#         # Print the keywords used (so you can verify no stopwords appear)
#         module_b = output.get("module_b", {})
#         print("\nUsed EN keywords:", module_b.get("retrieval_keywords", {}).get("en", []))
#         print("Used BN keywords:", module_b.get("retrieval_keywords", {}).get("bn", []))

#         # Print results for each model simultaneously (EN + BN)
#         for model_name in ["bm25", "tfidf", "fuzzy", "semantic", "hybrid"]:
#             if model_name in output.get("en", {}):
#                 _print_results_block("EN", model_name, output["en"][model_name])
#             if model_name in output.get("bn", {}):
#                 _print_results_block("BN", model_name, output["bn"][model_name])

#         print("\nTiming (ms):", output["timings_ms"])


# backend/clir/query_retrieval.py
"""
Module C — Retrieval Models (Core)

This file:
- Calls Module B QueryProcessor to get:
    - retrieval_queries  ✅ (clean queries for retrieval)
    - retrieval_keywords ✅ (important keywords only; stopwords removed)
- Loads your dataset from:
    backend/data/processed/en.jsonl
    backend/data/processed/bn.jsonl
- Implements and compares retrieval models:
  Model 1: Lexical Retrieval (TF-IDF + BM25)
  Model 2: Fuzzy/Transliteration Matching (RapidFuzz or difflib fallback)
  Model 3: Semantic Matching (sentence-transformers embeddings)
  Model 4: Hybrid Ranking (weighted fusion)

✅ FIX INCLUDED:
- Evidence matching now uses ONLY `retrieval_keywords` from Module B.
- Words like: what/where/are/the/and will NOT be used for matching (because Module B removed them).
- Also uses word-boundary matching for English (so partial matches don't trigger).

Run:
  cd backend
  python -m clir.query_retrieval
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import os
import json
import math
import time
import re

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import minmax_scale

from clir.query_processor import QueryProcessor, QueryProcessingResult


# -----------------------------
# Paths / Config
# -----------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)

DEFAULT_BN_JSONL_PATH = os.path.join(BACKEND_DIR, "data", "processed", "bn.jsonl")
DEFAULT_EN_JSONL_PATH = os.path.join(BACKEND_DIR, "data", "processed", "en.jsonl")

DEFAULT_EMBEDDING_CACHE_DIR = os.path.join(BACKEND_DIR, "models", "embeddings_cache")
os.makedirs(DEFAULT_EMBEDDING_CACHE_DIR, exist_ok=True)

_WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)


def _normalize_text_for_indexing(text: str) -> str:
    text = (text or "").strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


# -----------------------------
# Data Structures
# -----------------------------

@dataclass
class DocumentRecord:
    doc_id: int
    language: str  # 'bn' or 'en'
    title: str
    body: str
    url: str
    date: Optional[str] = None

    @property
    def full_text(self) -> str:
        title = _normalize_text_for_indexing(self.title)
        body = _normalize_text_for_indexing(self.body)
        return f"{title}. {title}. {body}".strip()


@dataclass
class ScoredResult:
    doc_id: int
    language: str
    url: str
    title: str
    date: Optional[str]
    score: float
    model: str

    # ✅ Evidence fields for debugging/demo
    matched_keywords: List[str]
    evidence_lines: List[str]


# -----------------------------
# JSONL Loader
# -----------------------------

def load_jsonl_documents(jsonl_path: str, language_code: str, starting_doc_id: int = 0) -> List[DocumentRecord]:
    documents: List[DocumentRecord] = []
    doc_id_counter = starting_doc_id

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

    with open(jsonl_path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            title = str(record.get("title", "") or "")
            body = str(record.get("body", "") or "")
            url = str(record.get("url", "") or "")
            date = record.get("date", None)

            if not (title or body):
                continue

            documents.append(
                DocumentRecord(
                    doc_id=doc_id_counter,
                    language=language_code,
                    title=title,
                    body=body,
                    url=url,
                    date=date,
                )
            )
            doc_id_counter += 1

    return documents


# -----------------------------
# Model 1: TF-IDF
# -----------------------------

class TfidfRetriever:
    def __init__(self, documents: List[DocumentRecord]) -> None:
        self.documents = documents
        self.vectorizer = TfidfVectorizer(
            lowercase=False,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
        )
        self.document_matrix = None  # scipy sparse

    def build(self) -> None:
        corpus = [doc.full_text for doc in self.documents]
        self.document_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query_text: str, top_k: int = 10) -> Tuple[List[int], np.ndarray]:
        if self.document_matrix is None:
            raise RuntimeError("TF-IDF retriever not built. Call build() first.")

        query_vector = self.vectorizer.transform([query_text])
        scores = (self.document_matrix @ query_vector.T).toarray().reshape(-1)

        if top_k <= 0:
            top_k = 10

        top_indices = np.argsort(-scores)[:top_k]
        return top_indices.tolist(), scores


# -----------------------------
# Model 1: BM25 (Okapi)
# -----------------------------

class BM25Retriever:
    def __init__(self, documents: List[DocumentRecord], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b

        self.tokenized_documents: List[List[str]] = []
        self.term_document_frequency: Dict[str, int] = {}
        self.document_term_frequencies: List[Dict[str, int]] = []
        self.document_lengths: List[int] = []
        self.average_document_length: float = 0.0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = _normalize_text_for_indexing(text)
        return [t for t in text.split(" ") if t]

    def build(self) -> None:
        self.tokenized_documents = []
        self.term_document_frequency = {}
        self.document_term_frequencies = []
        self.document_lengths = []

        for doc in self.documents:
            tokens = self._tokenize(doc.full_text)
            self.tokenized_documents.append(tokens)
            self.document_lengths.append(len(tokens))

            term_counts: Dict[str, int] = {}
            unique_terms_in_doc = set()

            for term in tokens:
                term_counts[term] = term_counts.get(term, 0) + 1
                unique_terms_in_doc.add(term)

            self.document_term_frequencies.append(term_counts)

            for term in unique_terms_in_doc:
                self.term_document_frequency[term] = self.term_document_frequency.get(term, 0) + 1

        total_length = sum(self.document_lengths) if self.document_lengths else 0
        self.average_document_length = (total_length / len(self.documents)) if self.documents else 0.0

    def _idf(self, term: str) -> float:
        document_count = len(self.documents)
        doc_freq = self.term_document_frequency.get(term, 0)
        return math.log(1 + (document_count - doc_freq + 0.5) / (doc_freq + 0.5))

    def search(self, query_text: str, top_k: int = 10) -> Tuple[List[int], np.ndarray]:
        query_tokens = self._tokenize(query_text)
        scores = np.zeros(len(self.documents), dtype=np.float32)

        for doc_index, term_counts in enumerate(self.document_term_frequencies):
            doc_length = self.document_lengths[doc_index] if self.document_lengths else 0
            length_norm = (1 - self.b) + self.b * (doc_length / (self.average_document_length + 1e-9))

            score = 0.0
            for term in query_tokens:
                tf = term_counts.get(term, 0)
                if tf == 0:
                    continue

                idf = self._idf(term)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * length_norm
                score += idf * (numerator / (denominator + 1e-9))

            scores[doc_index] = score

        top_indices = np.argsort(-scores)[:top_k]
        return top_indices.tolist(), scores


# -----------------------------
# Model 2: Fuzzy / Transliteration Matching
# -----------------------------

class FuzzyRetriever:
    def __init__(self, documents: List[DocumentRecord]) -> None:
        self.documents = documents

        self._use_rapidfuzz = False
        self._rapidfuzz_fuzz = None

        try:
            from rapidfuzz import fuzz as rapidfuzz_fuzz  # type: ignore
            self._use_rapidfuzz = True
            self._rapidfuzz_fuzz = rapidfuzz_fuzz
        except Exception:
            self._use_rapidfuzz = False

        self.document_search_strings: List[str] = []
        for doc in self.documents:
            preview_body = _normalize_text_for_indexing(doc.body)[:400]
            combined = f"{doc.title} {preview_body} {doc.url}"
            self.document_search_strings.append(combined)

    def search(self, query_text: str, top_k: int = 10) -> Tuple[List[int], np.ndarray]:
        query_text = _normalize_text_for_indexing(query_text)
        scores = np.zeros(len(self.documents), dtype=np.float32)

        if self._use_rapidfuzz and self._rapidfuzz_fuzz is not None:
            scorer = self._rapidfuzz_fuzz.token_sort_ratio
            for doc_index, search_text in enumerate(self.document_search_strings):
                similarity = scorer(query_text, search_text)  # 0..100
                scores[doc_index] = float(similarity) / 100.0
        else:
            import difflib
            for doc_index, search_text in enumerate(self.document_search_strings):
                similarity = difflib.SequenceMatcher(None, query_text, search_text).ratio()
                scores[doc_index] = float(similarity)

        top_indices = np.argsort(-scores)[:top_k]
        return top_indices.tolist(), scores


# -----------------------------
# Model 3: Semantic Matching (sentence-transformers)
# -----------------------------

class SemanticRetriever:
    def __init__(
        self,
        documents: List[DocumentRecord],
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_dir: str = DEFAULT_EMBEDDING_CACHE_DIR,
        cache_key: str = "default",
    ) -> None:
        self.documents = documents
        self.embedding_model_name = embedding_model_name
        self.cache_dir = cache_dir
        self.cache_key = cache_key

        self._model = None
        self.document_embeddings: Optional[np.ndarray] = None

        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_model(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer  # type: ignore
        self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def _cache_path(self) -> str:
        safe_key = re.sub(r"[^a-zA-Z0-9_\-]+", "_", self.cache_key)
        safe_model = re.sub(r"[^a-zA-Z0-9_\-]+", "_", self.embedding_model_name)
        return os.path.join(self.cache_dir, f"emb_{safe_model}_{safe_key}.npz")

    def build(self, force_rebuild: bool = False) -> None:
        cache_path = self._cache_path()

        if (not force_rebuild) and os.path.exists(cache_path):
            cached = np.load(cache_path)
            self.document_embeddings = cached["embeddings"]
            return

        embedding_model = self._load_model()
        document_texts = [doc.full_text for doc in self.documents]
        embeddings = embedding_model.encode(document_texts, convert_to_numpy=True, show_progress_bar=True)
        embeddings = embeddings.astype(np.float32)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        embeddings = embeddings / norms

        self.document_embeddings = embeddings
        np.savez_compressed(cache_path, embeddings=embeddings)

    def search(self, query_text: str, top_k: int = 10) -> Tuple[List[int], np.ndarray]:
        if self.document_embeddings is None:
            raise RuntimeError("Semantic retriever not built. Call build() first.")

        embedding_model = self._load_model()
        query_embedding = embedding_model.encode([query_text], convert_to_numpy=True).astype(np.float32)
        query_embedding = query_embedding / (np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-9)

        scores = (self.document_embeddings @ query_embedding.T).reshape(-1)
        top_indices = np.argsort(-scores)[:top_k]
        return top_indices.tolist(), scores


# -----------------------------
# Hybrid Fusion
# -----------------------------

def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    if np.all(scores == scores[0]):
        return np.zeros_like(scores, dtype=np.float32)
    return minmax_scale(scores.astype(np.float32))


def fuse_scores(
    bm25_scores: np.ndarray,
    tfidf_scores: np.ndarray,
    fuzzy_scores: np.ndarray,
    semantic_scores: np.ndarray,
    weights: Dict[str, float],
) -> np.ndarray:
    bm25_norm = _normalize_scores(bm25_scores)
    tfidf_norm = _normalize_scores(tfidf_scores)
    fuzzy_norm = _normalize_scores(fuzzy_scores)
    semantic_norm = _normalize_scores(semantic_scores)

    final_scores = (
        weights.get("bm25", 0.0) * bm25_norm
        + weights.get("tfidf", 0.0) * tfidf_norm
        + weights.get("fuzzy", 0.0) * fuzzy_norm
        + weights.get("semantic", 0.0) * semantic_norm
    )
    return final_scores.astype(np.float32)


# -----------------------------
# Evidence Matching (FIXED)
# -----------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?:\r?\n)+|(?<=[\.\!\?।])\s+", flags=re.UNICODE)

def _build_english_keyword_regex(keyword: str) -> re.Pattern:
    escaped = re.escape(keyword.lower())
    # word boundary for English-like tokens
    return re.compile(rf"\b{escaped}\b", flags=re.IGNORECASE)

def find_evidence_lines_for_document(
    document: DocumentRecord,
    language_code: str,
    keywords: List[str],
    max_lines: int = 3,
) -> Tuple[List[str], List[str]]:
    """
    Returns (matched_keywords, evidence_lines)
    ✅ Uses ONLY keywords provided (which come from Module B retrieval_keywords).
    ✅ For English uses word boundary matching.
    ✅ For Bangla uses substring match (works reasonably for BN script).
    """
    keywords = [k.strip() for k in (keywords or []) if k and k.strip()]
    if not keywords:
        return [], []

    title_text = (document.title or "")
    body_text = (document.body or "")
    combined_text = f"{title_text}\n{body_text}"

    matched_keywords: List[str] = []

    if language_code == "en":
        keyword_patterns = [(kw, _build_english_keyword_regex(kw)) for kw in keywords]
        lower_title = title_text.lower()
        lower_body = body_text.lower()

        for kw, pattern in keyword_patterns:
            if pattern.search(lower_title) or pattern.search(lower_body):
                matched_keywords.append(kw)
    else:
        # BN/mixed: simple substring
        for kw in keywords:
            if kw in title_text or kw in body_text:
                matched_keywords.append(kw)

    if not matched_keywords:
        return [], []

    # Split into "sentences/lines" for evidence
    raw_parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(combined_text) if p and p.strip()]
    evidence_lines: List[str] = []

    if language_code == "en":
        patterns = [_build_english_keyword_regex(kw) for kw in matched_keywords]
        for part in raw_parts:
            part_lower = part.lower()
            if any(p.search(part_lower) for p in patterns):
                evidence_lines.append(part)
            if len(evidence_lines) >= max_lines:
                break
    else:
        for part in raw_parts:
            if any(kw in part for kw in matched_keywords):
                evidence_lines.append(part)
            if len(evidence_lines) >= max_lines:
                break

    return matched_keywords, evidence_lines


# -----------------------------
# Query Retrieval Engine (Module C)
# -----------------------------

class QueryRetrievalEngine:
    """
    Models supported:
      - "bm25"
      - "tfidf"
      - "fuzzy"
      - "semantic"
      - "hybrid"
      - "all"
    """

    def __init__(
        self,
        bangla_jsonl_path: str = DEFAULT_BN_JSONL_PATH,
        english_jsonl_path: str = DEFAULT_EN_JSONL_PATH,
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        enable_stopwords: bool = True,
    ) -> None:
        self.bangla_jsonl_path = bangla_jsonl_path
        self.english_jsonl_path = english_jsonl_path
        self.embedding_model_name = embedding_model_name

        self.query_processor = QueryProcessor(enable_stopwords=enable_stopwords)

        self.documents_bn: List[DocumentRecord] = []
        self.documents_en: List[DocumentRecord] = []

        self.tfidf_bn: Optional[TfidfRetriever] = None
        self.tfidf_en: Optional[TfidfRetriever] = None

        self.bm25_bn: Optional[BM25Retriever] = None
        self.bm25_en: Optional[BM25Retriever] = None

        self.fuzzy_bn: Optional[FuzzyRetriever] = None
        self.fuzzy_en: Optional[FuzzyRetriever] = None

        self.semantic_bn: Optional[SemanticRetriever] = None
        self.semantic_en: Optional[SemanticRetriever] = None

        self._load_documents()
        self._build_retrievers()

    def _load_documents(self) -> None:
        self.documents_bn = load_jsonl_documents(self.bangla_jsonl_path, "bn", starting_doc_id=0)
        self.documents_en = load_jsonl_documents(self.english_jsonl_path, "en", starting_doc_id=len(self.documents_bn))

    def _build_retrievers(self) -> None:
        self.tfidf_bn = TfidfRetriever(self.documents_bn)
        self.tfidf_bn.build()

        self.tfidf_en = TfidfRetriever(self.documents_en)
        self.tfidf_en.build()

        self.bm25_bn = BM25Retriever(self.documents_bn)
        self.bm25_bn.build()

        self.bm25_en = BM25Retriever(self.documents_en)
        self.bm25_en.build()

        self.fuzzy_bn = FuzzyRetriever(self.documents_bn)
        self.fuzzy_en = FuzzyRetriever(self.documents_en)

        self.semantic_bn = SemanticRetriever(
            self.documents_bn,
            embedding_model_name=self.embedding_model_name,
            cache_key="bn",
        )
        self.semantic_bn.build(force_rebuild=False)

        self.semantic_en = SemanticRetriever(
            self.documents_en,
            embedding_model_name=self.embedding_model_name,
            cache_key="en",
        )
        self.semantic_en.build(force_rebuild=False)

    def _format_results(
        self,
        documents: List[DocumentRecord],
        scores: np.ndarray,
        top_indices: List[int],
        model_name: str,
        language_code: str,
        retrieval_keywords: List[str],
    ) -> List[ScoredResult]:
        results: List[ScoredResult] = []
        for doc_index in top_indices:
            doc = documents[doc_index]

            matched_keywords, evidence_lines = find_evidence_lines_for_document(
                document=doc,
                language_code=language_code,
                keywords=retrieval_keywords,
                max_lines=3,
            )

            results.append(
                ScoredResult(
                    doc_id=doc.doc_id,
                    language=language_code,
                    url=doc.url,
                    title=doc.title,
                    date=doc.date,
                    score=float(scores[doc_index]),
                    model=model_name,
                    matched_keywords=matched_keywords,
                    evidence_lines=evidence_lines,
                )
            )
        return results

    def _search_single_language(
        self,
        language_code: str,
        retrieval_queries: List[str],
        retrieval_keywords: List[str],
        top_k: int,
        model: str,
        hybrid_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, List[ScoredResult]]:
        if language_code == "bn":
            documents = self.documents_bn
            tfidf_retriever = self.tfidf_bn
            bm25_retriever = self.bm25_bn
            fuzzy_retriever = self.fuzzy_bn
            semantic_retriever = self.semantic_bn
        else:
            documents = self.documents_en
            tfidf_retriever = self.tfidf_en
            bm25_retriever = self.bm25_en
            fuzzy_retriever = self.fuzzy_en
            semantic_retriever = self.semantic_en

        if tfidf_retriever is None or bm25_retriever is None or fuzzy_retriever is None or semantic_retriever is None:
            raise RuntimeError("Retrievers not initialized properly.")

        def max_over_variants(search_fn):
            best_scores = np.zeros(len(documents), dtype=np.float32)
            for query_text in retrieval_queries:
                _, scores = search_fn(query_text, top_k=len(documents))
                best_scores = np.maximum(best_scores, scores.astype(np.float32))
            return best_scores

        outputs: Dict[str, List[ScoredResult]] = {}

        tfidf_scores = max_over_variants(tfidf_retriever.search) if model in ("tfidf", "all", "hybrid") else np.zeros(len(documents), dtype=np.float32)
        bm25_scores = max_over_variants(bm25_retriever.search) if model in ("bm25", "all", "hybrid") else np.zeros(len(documents), dtype=np.float32)
        fuzzy_scores = max_over_variants(fuzzy_retriever.search) if model in ("fuzzy", "all", "hybrid") else np.zeros(len(documents), dtype=np.float32)
        semantic_scores = max_over_variants(semantic_retriever.search) if model in ("semantic", "all", "hybrid") else np.zeros(len(documents), dtype=np.float32)

        if model in ("tfidf", "all"):
            top_indices = np.argsort(-tfidf_scores)[:top_k].tolist()
            outputs["tfidf"] = self._format_results(documents, tfidf_scores, top_indices, "tfidf", language_code, retrieval_keywords)

        if model in ("bm25", "all"):
            top_indices = np.argsort(-bm25_scores)[:top_k].tolist()
            outputs["bm25"] = self._format_results(documents, bm25_scores, top_indices, "bm25", language_code, retrieval_keywords)

        if model in ("fuzzy", "all"):
            top_indices = np.argsort(-fuzzy_scores)[:top_k].tolist()
            outputs["fuzzy"] = self._format_results(documents, fuzzy_scores, top_indices, "fuzzy", language_code, retrieval_keywords)

        if model in ("semantic", "all"):
            top_indices = np.argsort(-semantic_scores)[:top_k].tolist()
            outputs["semantic"] = self._format_results(documents, semantic_scores, top_indices, "semantic", language_code, retrieval_keywords)

        if model in ("hybrid", "all"):
            if hybrid_weights is None:
                hybrid_weights = {"bm25": 0.30, "tfidf": 0.10, "fuzzy": 0.20, "semantic": 0.40}

            hybrid_scores = fuse_scores(
                bm25_scores=bm25_scores,
                tfidf_scores=tfidf_scores,
                fuzzy_scores=fuzzy_scores,
                semantic_scores=semantic_scores,
                weights=hybrid_weights,
            )
            top_indices = np.argsort(-hybrid_scores)[:top_k].tolist()
            outputs["hybrid"] = self._format_results(documents, hybrid_scores, top_indices, "hybrid", language_code, retrieval_keywords)

        return outputs

    def search(
        self,
        user_query: str,
        top_k: int = 10,
        model: str = "all",
        hybrid_weights: Optional[Dict[str, float]] = None,
        include_debug: bool = True,
    ) -> Dict[str, Any]:
        if model not in ("bm25", "tfidf", "fuzzy", "semantic", "hybrid", "all"):
            raise ValueError("model must be one of: bm25|tfidf|fuzzy|semantic|hybrid|all")

        start_time = time.perf_counter()

        module_b_result: QueryProcessingResult = self.query_processor.process(user_query)
        module_b_dict = module_b_result.to_dict()

        # ✅ FIX: Use ONLY retrieval_queries from Module B for retrieval
        retrieval_queries_bn = module_b_dict.get("retrieval_queries", {}).get("bn", []) or []
        retrieval_queries_en = module_b_dict.get("retrieval_queries", {}).get("en", []) or []

        # ✅ FIX: Use ONLY retrieval_keywords from Module B for evidence matching
        retrieval_keywords_bn = module_b_dict.get("retrieval_keywords", {}).get("bn", []) or []
        retrieval_keywords_en = module_b_dict.get("retrieval_keywords", {}).get("en", []) or []

        # Fallback safety (should rarely happen)
        if not retrieval_queries_bn:
            retrieval_queries_bn = module_b_result.expanded_queries.get("bn", [])[:1] or [module_b_result.normalized_query]
        if not retrieval_queries_en:
            retrieval_queries_en = module_b_result.expanded_queries.get("en", [])[:1] or [module_b_result.normalized_query]

        module_b_time = time.perf_counter()

        bn_results = self._search_single_language(
            language_code="bn",
            retrieval_queries=retrieval_queries_bn[:5],
            retrieval_keywords=retrieval_keywords_bn,
            top_k=top_k,
            model=model,
            hybrid_weights=hybrid_weights,
        )
        bn_time = time.perf_counter()

        en_results = self._search_single_language(
            language_code="en",
            retrieval_queries=retrieval_queries_en[:5],
            retrieval_keywords=retrieval_keywords_en,
            top_k=top_k,
            model=model,
            hybrid_weights=hybrid_weights,
        )
        en_time = time.perf_counter()

        timings_ms = {
            "module_b_processing": (module_b_time - start_time) * 1000.0,
            "bn_retrieval": (bn_time - module_b_time) * 1000.0,
            "en_retrieval": (en_time - bn_time) * 1000.0,
            "total": (en_time - start_time) * 1000.0,
        }

        response: Dict[str, Any] = {
            "bn": {k: [r.__dict__ for r in v] for k, v in bn_results.items()},
            "en": {k: [r.__dict__ for r in v] for k, v in en_results.items()},
            "timings_ms": timings_ms,
        }
        if include_debug:
            response["module_b"] = module_b_dict

        return response


# -----------------------------
# Minimal CLI test
# -----------------------------

def _print_results_block(language_label: str, model_name: str, results: List[Dict[str, Any]]) -> None:
    print(f"\n--- {language_label} | {model_name.upper()} ---")
    for rank, item in enumerate(results, start=1):
        print(f"{rank}. score={item['score']:.4f} | {item['title']}")
        print(f"    url: {item['url']}")
        matched = item.get("matched_keywords", []) or []
        if matched:
            print(f"    matched keywords: {', '.join(matched)}")
        evidence = item.get("evidence_lines", []) or []
        for line in evidence:
            print(f"    line: {line}")


if __name__ == "__main__":
    retrieval_engine = QueryRetrievalEngine()

    while True:
        user_query = input("Query (empty to quit): ").strip()
        if not user_query:
            break

        output = retrieval_engine.search(user_query, top_k=5, model="all", include_debug=True)

        # Print the keywords used (so you can verify no stopwords appear)
        module_b = output.get("module_b", {})
        print("\nUsed EN keywords:", module_b.get("retrieval_keywords", {}).get("en", []))
        print("Used BN keywords:", module_b.get("retrieval_keywords", {}).get("bn", []))

        # Print results for each model simultaneously (EN + BN)
        for model_name in ["bm25", "tfidf", "fuzzy", "semantic", "hybrid"]:
            if model_name in output.get("en", {}):
                _print_results_block("EN", model_name, output["en"][model_name])
            if model_name in output.get("bn", {}):
                _print_results_block("BN", model_name, output["bn"][model_name])

        print("\nTiming (ms):", output["timings_ms"])