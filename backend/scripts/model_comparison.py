"""
Model Comparison Tool for Module C

Compares BM25 vs TF-IDF, analyzes failure cases for synonyms/paraphrases,
and compares all retrieval models.
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any, Set
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from clir.evaluation import RankingAndScoringEngine


class ModelComparator:
    """Compares different retrieval models and analyzes failure cases."""
    
    def __init__(self):
        self.ranking_engine = RankingAndScoringEngine()
    
    def compare_bm25_vs_tfidf(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Compare BM25 and TF-IDF performance."""
        bm25_result = self.ranking_engine.rank(
            user_query=query,
            model_name="bm25",
            top_k=top_k
        )
        
        tfidf_result = self.ranking_engine.rank(
            user_query=query,
            model_name="tfidf",
            top_k=top_k
        )
        
        bm25_urls = {doc.url for doc in bm25_result.ranked_documents}
        tfidf_urls = {doc.url for doc in tfidf_result.ranked_documents}
        
        overlap = len(bm25_urls & tfidf_urls)
        bm25_only = bm25_urls - tfidf_urls
        tfidf_only = tfidf_urls - bm25_urls
        
        return {
            "query": query,
            "bm25_top_k": [{"title": doc.title, "url": doc.url, "score": doc.raw_score} 
                          for doc in bm25_result.ranked_documents],
            "tfidf_top_k": [{"title": doc.title, "url": doc.url, "score": doc.raw_score} 
                           for doc in tfidf_result.ranked_documents],
            "overlap_count": overlap,
            "bm25_only_count": len(bm25_only),
            "tfidf_only_count": len(tfidf_only),
            "bm25_only_urls": list(bm25_only),
            "tfidf_only_urls": list(tfidf_only),
            "analysis": self._analyze_lexical_differences(bm25_result, tfidf_result, overlap)
        }
    
    def _analyze_lexical_differences(self, bm25_result, tfidf_result, overlap: int) -> str:
        """Analyze why BM25 and TF-IDF differ."""
        analysis = []
        
        if overlap < len(bm25_result.ranked_documents) * 0.5:
            analysis.append(
                f"Low overlap ({overlap}/{len(bm25_result.ranked_documents)}) suggests different ranking strategies. "
                f"BM25 uses term frequency normalization and document length penalization, "
                f"while TF-IDF emphasizes rare terms. This difference is expected for queries with "
                f"varying term frequencies across the corpus."
            )
        
        # Check if one model has higher confidence scores
        bm25_avg_score = sum(doc.raw_score for doc in bm25_result.ranked_documents[:5]) / 5 if bm25_result.ranked_documents else 0
        tfidf_avg_score = sum(doc.raw_score for doc in tfidf_result.ranked_documents[:5]) / 5 if tfidf_result.ranked_documents else 0
        
        if abs(bm25_avg_score - tfidf_avg_score) > 0.1:
            analysis.append(
                f"Score difference: BM25 avg={bm25_avg_score:.3f}, TF-IDF avg={tfidf_avg_score:.3f}. "
                f"BM25 typically performs better for longer queries and documents with varying lengths."
            )
        
        return " ".join(analysis) if analysis else "Both models show similar results."
    
    def analyze_failure_cases(self, query: str) -> Dict[str, Any]:
        """Analyze failure cases for synonyms, paraphrases, and cross-script terms."""
        failures = {
            "synonym_failures": [],
            "paraphrase_failures": [],
            "cross_script_failures": []
        }
        
        # Test with lexical models
        lexical_result = self.ranking_engine.rank(
            user_query=query,
            model_name="bm25",
            top_k=10
        )
        
        # Test with semantic model
        semantic_result = self.ranking_engine.rank(
            user_query=query,
            model_name="semantic",
            top_k=10
        )
        
        lexical_urls = {doc.url for doc in lexical_result.ranked_documents[:5]}
        semantic_urls = {doc.url for doc in semantic_result.ranked_documents[:5]}
        
        semantic_only = semantic_urls - lexical_urls
        
        if semantic_only:
            failures["synonym_failures"].append({
                "query": query,
                "issue": "Lexical models (BM25/TF-IDF) fail to match synonyms or paraphrases",
                "lexical_results": len(lexical_urls),
                "semantic_results": len(semantic_urls),
                "semantic_only_urls": list(semantic_only)[:3],
                "explanation": "Lexical models rely on exact term matching. If a document uses synonyms "
                              "or paraphrases (e.g., 'education' vs 'learning', 'school'), lexical models "
                              "may miss relevant documents that semantic models can find."
            })
        
        # Check for cross-script issues
        # If query is in one script but documents are in another
        query_has_bengali = any(ord(c) >= 0x0980 and ord(c) <= 0x09FF for c in query)
        query_has_english = any(("A" <= c <= "Z") or ("a" <= c <= "z") for c in query)
        
        if query_has_bengali or query_has_english:
            bn_docs = [doc for doc in lexical_result.ranked_documents if doc.language == "bn"]
            en_docs = [doc for doc in lexical_result.ranked_documents if doc.language == "en"]
            
            if query_has_bengali and len(en_docs) > len(bn_docs):
                failures["cross_script_failures"].append({
                    "query": query,
                    "issue": "Bengali query retrieved more English documents",
                    "bn_results": len(bn_docs),
                    "en_results": len(en_docs),
                    "explanation": "Lexical models cannot match across scripts without translation. "
                                  "A Bengali query like 'ঢাকা' cannot match English documents containing 'Dhaka' "
                                  "without proper transliteration or translation."
                })
        
        return failures
    
    def compare_all_models(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Compare all models: BM25, TF-IDF, Fuzzy, Semantic, Hybrid."""
        models = ["bm25", "tfidf", "fuzzy", "semantic", "hybrid"]
        results = {}
        
        for model_name in models:
            result = self.ranking_engine.rank(
                user_query=query,
                model_name=model_name,
                top_k=top_k
            )
            results[model_name] = {
                "top_5": [{"title": doc.title, "url": doc.url, "confidence": doc.matching_confidence} 
                         for doc in result.ranked_documents[:5]],
                "top_confidence": result.top_confidence,
                "warning": result.warning_low_confidence
            }
        
        # Calculate overlap between models
        model_urls = {model: {doc["url"] for doc in results[model]["top_5"]} for model in models}
        
        overlaps = {}
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                overlap = len(model_urls[model1] & model_urls[model2])
                overlaps[f"{model1}_vs_{model2}"] = overlap
        
        return {
            "query": query,
            "model_results": results,
            "overlaps": overlaps,
            "recommendation": self._recommend_model(results, overlaps)
        }
    
    def _recommend_model(self, results: Dict[str, Any], overlaps: Dict[str, int]) -> str:
        """Recommend which model to use based on comparison."""
        # Check if hybrid has best confidence
        hybrid_conf = results.get("hybrid", {}).get("top_confidence", 0)
        semantic_conf = results.get("semantic", {}).get("top_confidence", 0)
        lexical_conf = max(
            results.get("bm25", {}).get("top_confidence", 0),
            results.get("tfidf", {}).get("top_confidence", 0)
        )
        
        if hybrid_conf > 0.5:
            return "Hybrid model recommended: combines strengths of lexical and semantic models."
        elif semantic_conf > lexical_conf + 0.2:
            return "Semantic model recommended: better for cross-lingual and conceptual queries."
        else:
            return "Lexical model (BM25) recommended: good for exact keyword matching."


def generate_comparison_report(comparisons: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Module C — Model Comparison Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## 1. BM25 vs TF-IDF Comparison\n\n")
        f.write("### Analysis\n\n")
        f.write("Both BM25 and TF-IDF are lexical retrieval models, but they differ in their ranking strategies:\n\n")
        f.write("- **BM25**: Uses term frequency normalization with document length penalization. "
                "Better for longer documents and queries with repeated terms.\n")
        f.write("- **TF-IDF**: Emphasizes rare terms (high IDF). Better for distinguishing documents "
                "with unique vocabulary.\n\n")
        
        for comp in comparisons:
            if "bm25_top_k" in comp:
                f.write(f"### Query: `{comp['query']}`\n\n")
                f.write(f"- **Overlap**: {comp['overlap_count']}/10 documents\n")
                f.write(f"- **BM25-only**: {comp['bm25_only_count']} documents\n")
                f.write(f"- **TF-IDF-only**: {comp['tfidf_only_count']} documents\n\n")
                f.write(f"**Analysis**: {comp['analysis']}\n\n")
                f.write("---\n\n")
        
        f.write("## 2. Failure Case Analysis\n\n")
        f.write("### Why Lexical Models Fail\n\n")
        f.write("#### Synonyms and Paraphrases\n\n")
        f.write("Lexical models (BM25, TF-IDF) fail when:\n")
        f.write("- Documents use synonyms (e.g., 'education' vs 'learning')\n")
        f.write("- Documents use paraphrases (e.g., 'good news' vs 'positive developments')\n")
        f.write("- Query and document use different terminology\n\n")
        f.write("**Solution**: Use semantic models or query expansion.\n\n")
        
        f.write("#### Cross-Script Terms\n\n")
        f.write("Lexical models fail when:\n")
        f.write("- Query is in Bengali but documents are in English (e.g., 'ঢাকা' vs 'Dhaka')\n")
        f.write("- Query is in English but documents are in Bengali\n")
        f.write("- No transliteration or translation mapping exists\n\n")
        f.write("**Solution**: Use translation (Module B), transliteration matching (Fuzzy model), "
                "or semantic models (multilingual embeddings).\n\n")
        
        f.write("## 3. Model Recommendations\n\n")
        f.write("| Use Case | Recommended Model | Reason |\n")
        f.write("|----------|-------------------|--------|\n")
        f.write("| Exact keyword matching | BM25 | Best for precise term matching |\n")
        f.write("| Cross-lingual queries | Semantic | Multilingual embeddings handle translation |\n")
        f.write("| Mixed queries | Hybrid | Combines lexical + semantic signals |\n")
        f.write("| Transliteration needed | Fuzzy + Semantic | Fuzzy for edit distance, Semantic for meaning |\n")
        f.write("| General purpose | Hybrid | Best overall performance |\n\n")
    
    print(f"Comparison report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Model Comparison Tool for Module C")
    parser.add_argument(
        "--queries",
        type=str,
        required=True,
        help="Path to text file with test queries (one per line)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/eval/model_comparison_report.md",
        help="Output markdown report path",
    )
    parser.add_argument(
        "--analysis",
        type=str,
        choices=["all", "bm25-tfidf", "failures", "all-models"],
        default="all",
        help="Type of analysis to perform",
    )
    
    args = parser.parse_args()
    
    # Load queries
    queries = []
    with open(args.queries, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)
    
    if not queries:
        print("Error: No queries found in file.")
        return
    
    print(f"Comparing models on {len(queries)} queries...\n")
    
    comparator = ModelComparator()
    all_comparisons = []
    
    for query in queries:
        
        if args.analysis in ("all", "bm25-tfidf"):
            comp = comparator.compare_bm25_vs_tfidf(query)
            all_comparisons.append(comp)
        
        if args.analysis in ("all", "failures"):
            failures = comparator.analyze_failure_cases(query)
            if failures["synonym_failures"] or failures["cross_script_failures"]:
                all_comparisons.append({"query": query, "failures": failures})
        
        if args.analysis in ("all", "all-models"):
            comp = comparator.compare_all_models(query)
            all_comparisons.append(comp)
    
    # Generate report
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_comparison_report(all_comparisons, args.output)
    
    print(f"\nComparison complete. Report: {args.output}")


if __name__ == "__main__":
    main()
