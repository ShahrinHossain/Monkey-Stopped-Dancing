"""
Error Analysis Tool for CLIR System

Analyzes retrieval failures across translation, NER, semantic/lexical,
cross-script, and code-switching categories.
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from clir.query_retrieval import QueryRetrievalEngine
from clir.query_processor import QueryProcessor
from clir.evaluation import RankingAndScoringEngine


class ErrorAnalyzer:
    """Analyzes retrieval errors and successes."""
    
    def __init__(self):
        self.retrieval_engine = QueryRetrievalEngine()
        self.query_processor = QueryProcessor()
        self.ranking_engine = RankingAndScoringEngine()
    
    def analyze_translation_failures(self, query: str) -> Optional[Dict[str, Any]]:
        """Analyze translation-related failures."""
        result = self.query_processor.process(query)
        
        # Check if translation happened
        if not result.translated_query:
            return None
        
        # Compare original and translated query
        original_lang = result.detected_language
        translated_query = result.translated_query
        
        # Get retrieval results for both original and translated
        original_results = self.ranking_engine.rank(
            user_query=query,
            model_name="hybrid",
            top_k=10
        )
        
        translated_results = self.ranking_engine.rank(
            user_query=translated_query,
            model_name="hybrid",
            top_k=10
        )
        
        # Check if translation changed meaning significantly
        original_top_urls = {doc.url for doc in original_results.ranked_documents[:5]}
        translated_top_urls = {doc.url for doc in translated_results.ranked_documents[:5]}
        
        overlap = len(original_top_urls & translated_top_urls)
        
        if overlap < 2:  # Low overlap suggests translation issue
            return {
                "category": "Translation Failure",
                "query": query,
                "original_language": original_lang,
                "translated_query": translated_query,
                "original_top_5": [doc.title for doc in original_results.ranked_documents[:5]],
                "translated_top_5": [doc.title for doc in translated_results.ranked_documents[:5]],
                "overlap_count": overlap,
                "analysis": f"Translation from {original_lang} to {'en' if original_lang == 'bn' else 'bn'} "
                           f"resulted in different top results (overlap: {overlap}/5). "
                           f"This suggests the translation may have changed the query meaning.",
                "recommendation": "Review translation quality. Consider using multiple translation backends or "
                               "query expansion to handle translation ambiguities."
            }
        
        return None
    
    def analyze_ner_mismatch(self, query: str) -> Optional[Dict[str, Any]]:
        """Analyze Named Entity Recognition and matching issues."""
        result = self.query_processor.process(query)
        
        # Extract named entities
        named_entities = result.named_entities
        mapped_entities = result.mapped_entities
        
        if not named_entities:
            return None
        
        # Get retrieval results
        retrieval_result = self.ranking_engine.rank(
            user_query=query,
            model_name="hybrid",
            top_k=10
        )
        
        # Check if top results contain the named entities
        top_docs = retrieval_result.ranked_documents[:5]
        entity_found_count = 0
        
        for doc in top_docs:
            doc_text = f"{doc.title} {' '.join(doc.matched_keywords)}"
            for entity in named_entities:
                if entity.lower() in doc_text.lower():
                    entity_found_count += 1
                    break
        
        if entity_found_count < 2:  # Entities not found in top results
            return {
                "category": "Named Entity Mismatch",
                "query": query,
                "detected_entities": named_entities,
                "mapped_entities": mapped_entities,
                "top_5_results": [
                    {
                        "title": doc.title,
                        "url": doc.url,
                        "matched_keywords": doc.matched_keywords,
                        "contains_entity": any(
                            entity.lower() in (doc.title + " " + " ".join(doc.matched_keywords)).lower()
                            for entity in named_entities
                        )
                    }
                    for doc in top_docs
                ],
                "entity_found_in_top_5": entity_found_count,
                "analysis": f"Query contains named entities: {', '.join(named_entities)}, "
                           f"but only {entity_found_count}/5 top results contain these entities. "
                           f"This suggests NER mapping or entity matching may be failing.",
                "recommendation": "Improve named entity mapping between languages. "
                               "Consider using a more comprehensive entity dictionary or "
                               "entity-aware retrieval boosting."
            }
        
        return None
    
    def analyze_semantic_vs_lexical(self, query: str) -> Optional[Dict[str, Any]]:
        """Compare semantic vs lexical retrieval models."""
        # Get results from different models
        lexical_result = self.ranking_engine.rank(
            user_query=query,
            model_name="bm25",
            top_k=10
        )
        
        semantic_result = self.ranking_engine.rank(
            user_query=query,
            model_name="semantic",
            top_k=10
        )
        
        hybrid_result = self.ranking_engine.rank(
            user_query=query,
            model_name="hybrid",
            top_k=10
        )
        
        lexical_urls = {doc.url for doc in lexical_result.ranked_documents[:5]}
        semantic_urls = {doc.url for doc in semantic_result.ranked_documents[:5]}
        hybrid_urls = {doc.url for doc in hybrid_result.ranked_documents[:5]}
        
        # Check if semantic found results that lexical didn't
        semantic_only = semantic_urls - lexical_urls
        
        if len(semantic_only) >= 2 and len(lexical_urls) < 3:
            return {
                "category": "Semantic vs. Lexical Wins",
                "query": query,
                "lexical_top_5": [doc.title for doc in lexical_result.ranked_documents[:5]],
                "semantic_top_5": [doc.title for doc in semantic_result.ranked_documents[:5]],
                "hybrid_top_5": [doc.title for doc in hybrid_result.ranked_documents[:5]],
                "lexical_count": len(lexical_urls),
                "semantic_count": len(semantic_urls),
                "semantic_only_urls": list(semantic_only),
                "analysis": f"Lexical model (BM25) found {len(lexical_urls)} unique results, "
                           f"while semantic model found {len(semantic_urls)} unique results. "
                           f"Semantic model found {len(semantic_only)} results not in lexical top-5. "
                           f"This demonstrates semantic matching can find relevant documents "
                           f"even when exact keyword matches are missing.",
                "recommendation": "Use hybrid model that combines lexical and semantic signals. "
                               "Semantic models are particularly useful for cross-lingual queries "
                               "and conceptual matching."
            }
        
        return None
    
    def analyze_cross_script_ambiguity(self, query: str) -> Optional[Dict[str, Any]]:
        """Analyze cross-script transliteration and ambiguity issues."""
        result = self.query_processor.process(query)
        
        # Check if query contains mixed scripts or transliteration
        has_bengali = any(ord(c) >= 0x0980 and ord(c) <= 0x09FF for c in query)
        has_english = any(("A" <= c <= "Z") or ("a" <= c <= "z") for c in query)
        
        if not (has_bengali and has_english):
            return None
        
        # Check retrieval results
        retrieval_result = self.ranking_engine.rank(
            user_query=query,
            model_name="hybrid",
            top_k=10
        )
        
        # Analyze if results match both scripts
        top_docs = retrieval_result.ranked_documents[:5]
        bn_matches = 0
        en_matches = 0
        
        for doc in top_docs:
            doc_text = f"{doc.title} {' '.join(doc.matched_keywords)}"
            if any(ord(c) >= 0x0980 and ord(c) <= 0x09FF for c in doc_text):
                bn_matches += 1
            if any(("A" <= c <= "Z") or ("a" <= c <= "z") for c in doc_text):
                en_matches += 1
        
        if bn_matches == 0 or en_matches == 0:
            return {
                "category": "Cross-Script Ambiguity",
                "query": query,
                "contains_bengali": has_bengali,
                "contains_english": has_english,
                "top_5_results": [
                    {
                        "title": doc.title,
                        "language": doc.language,
                        "matched_keywords": doc.matched_keywords
                    }
                    for doc in top_docs
                ],
                "bengali_matches": bn_matches,
                "english_matches": en_matches,
                "analysis": f"Query contains both Bengali and English scripts, but results show "
                           f"{bn_matches} Bengali matches and {en_matches} English matches. "
                           f"This suggests the system may not be handling cross-script queries optimally.",
                "recommendation": "Improve cross-script query handling. Consider transliteration "
                               "matching and script-aware query expansion."
            }
        
        return None
    
    def analyze_code_switching(self, query: str) -> Optional[Dict[str, Any]]:
        """Analyze code-switching (mixing languages in query)."""
        result = self.query_processor.process(query)
        
        # Check if query is mixed
        if result.detected_language != "mixed":
            return None
        
        # Get retrieval results
        retrieval_result = self.ranking_engine.rank(
            user_query=query,
            model_name="hybrid",
            top_k=10
        )
        
        # Check if results span both languages
        top_docs = retrieval_result.ranked_documents[:5]
        bn_docs = [doc for doc in top_docs if doc.language == "bn"]
        en_docs = [doc for doc in top_docs if doc.language == "en"]
        
        if len(bn_docs) == 0 or len(en_docs) == 0:
            return {
                "category": "Code-Switching",
                "query": query,
                "detected_language": result.detected_language,
                "normalized_query": result.normalized_query,
                "translated_query": result.translated_query,
                "top_5_results": [
                    {
                        "title": doc.title,
                        "language": doc.language,
                        "url": doc.url
                    }
                    for doc in top_docs
                ],
                "bengali_results": len(bn_docs),
                "english_results": len(en_docs),
                "analysis": f"Query mixes Bengali and English, but results are skewed: "
                           f"{len(bn_docs)} Bengali and {len(en_docs)} English documents in top-5. "
                           f"This suggests the system may not be optimally handling code-switched queries.",
                "recommendation": "Improve code-switching handling by: "
                               "1. Better language detection for mixed queries, "
                               "2. Query expansion in both languages, "
                               "3. Balanced retrieval from both language corpora."
            }
        
        return None
    
    def analyze_query(self, query: str) -> List[Dict[str, Any]]:
        """Run all analysis categories on a query."""
        findings = []
        
        # Translation failures
        finding = self.analyze_translation_failures(query)
        if finding:
            findings.append(finding)
        
        # NER mismatch
        finding = self.analyze_ner_mismatch(query)
        if finding:
            findings.append(finding)
        
        # Semantic vs lexical
        finding = self.analyze_semantic_vs_lexical(query)
        if finding:
            findings.append(finding)
        
        # Cross-script ambiguity
        finding = self.analyze_cross_script_ambiguity(query)
        if finding:
            findings.append(finding)
        
        # Code-switching
        finding = self.analyze_code_switching(query)
        if finding:
            findings.append(finding)
        
        return findings


def generate_report(findings: List[Dict[str, Any]], output_path: str) -> None:
    """Generate a markdown report from findings."""
    # Group by category
    by_category = {}
    for finding in findings:
        category = finding["category"]
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(finding)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# CLIR System Error Analysis Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"Total findings: {len(findings)}\n\n")
        for category, items in by_category.items():
            f.write(f"- **{category}**: {len(items)} case(s)\n")
        f.write("\n---\n\n")
        
        # Detailed findings by category
        for category in [
            "Translation Failure",
            "Named Entity Mismatch",
            "Semantic vs. Lexical Wins",
            "Cross-Script Ambiguity",
            "Code-Switching"
        ]:
            if category not in by_category:
                continue
            
            f.write(f"## {category}\n\n")
            
            for idx, finding in enumerate(by_category[category], 1):
                f.write(f"### Case Study {idx}\n\n")
                f.write(f"**Query**: `{finding['query']}`\n\n")
                
                if "original_language" in finding:
                    f.write(f"- **Original Language**: {finding['original_language']}\n")
                if "translated_query" in finding:
                    f.write(f"- **Translated Query**: `{finding['translated_query']}`\n")
                if "detected_entities" in finding:
                    f.write(f"- **Detected Entities**: {', '.join(finding['detected_entities'])}\n")
                
                f.write("\n**Retrieved Documents (Top 5)**:\n\n")
                if "top_5_results" in finding:
                    for i, doc in enumerate(finding["top_5_results"][:5], 1):
                        if isinstance(doc, dict):
                            f.write(f"{i}. **{doc.get('title', 'N/A')}**\n")
                            if "url" in doc:
                                f.write(f"   - URL: {doc['url']}\n")
                            if "language" in doc:
                                f.write(f"   - Language: {doc['language']}\n")
                            if "matched_keywords" in doc:
                                f.write(f"   - Keywords: {', '.join(doc['matched_keywords'][:5])}\n")
                        else:
                            f.write(f"{i}. {doc}\n")
                elif "original_top_5" in finding:
                    f.write("**Original Query Results**:\n")
                    for i, title in enumerate(finding["original_top_5"][:5], 1):
                        f.write(f"{i}. {title}\n")
                    f.write("\n**Translated Query Results**:\n")
                    for i, title in enumerate(finding["translated_top_5"][:5], 1):
                        f.write(f"{i}. {title}\n")
                
                f.write("\n**Analysis**:\n\n")
                f.write(f"{finding['analysis']}\n\n")
                
                if "recommendation" in finding:
                    f.write("**Recommendation**:\n\n")
                    f.write(f"{finding['recommendation']}\n\n")
                
                f.write("---\n\n")
    
    print(f"Report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Error Analysis Tool for CLIR System")
    parser.add_argument(
        "--queries",
        type=str,
        required=True,
        help="Path to text file with test queries (one per line)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/eval/error_analysis_report.md",
        help="Output markdown report path",
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["all", "translation", "ner", "semantic", "cross-script", "code-switching"],
        default="all",
        help="Specific category to analyze (default: all)",
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
    
    print(f"Analyzing {len(queries)} queries...\n")
    
    analyzer = ErrorAnalyzer()
    all_findings = []
    
    for query in queries:
        findings = analyzer.analyze_query(query)
        all_findings.extend(findings)
        if findings:
            print(f"Query: {query} - {len(findings)} issue(s) found")
    
    if not all_findings:
        print("\nNo issues found in the analyzed queries.")
        return
    
    # Generate report
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_report(all_findings, args.output)
    
    print(f"\nAnalysis complete. Total findings: {len(all_findings)}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
