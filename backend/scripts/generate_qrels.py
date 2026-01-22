"""
Generate qrels.jsonl file with real URLs from actual document retrieval.

This script runs queries and uses the top-k retrieved documents as "relevant"
to create a proper qrels file. You can manually edit it later to refine relevance.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from clir.evaluation import RankingAndScoringEngine


def load_queries(queries_path: str) -> List[str]:
    """Load queries from a text file (one per line, # for comments)."""
    queries = []
    if not Path(queries_path).exists():
        return queries
    
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)
    return queries


def generate_qrels(queries: List[str], output_path: str, top_k: int = 10) -> None:
    """Generate qrels file by running queries and using top-k results as relevant."""
    engine = RankingAndScoringEngine()
    
    qrels_entries = []
    
    print(f"Generating qrels for {len(queries)} queries...")
    print(f"Using top-{top_k} results per query as relevant URLs\n")
    
    for query in queries:
        print(f"Processing: {query}")
        result = engine.rank(
            user_query=query,
            model_name="hybrid",
            top_k=top_k,
            include_debug=False,
        )
        
        relevant_urls = [doc.url for doc in result.ranked_documents if doc.url]
        
        if relevant_urls:
            qrels_entries.append({
                "query": query,
                "relevant_urls": relevant_urls
            })
            print(f"  Found {len(relevant_urls)} documents")
        else:
            print(f"  WARNING: No documents retrieved for this query")
    
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in qrels_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"\n✓ Generated qrels file: {output_path}")
    print(f"  Total queries: {len(qrels_entries)}")
    print(f"  Total relevant URLs: {sum(len(e['relevant_urls']) for e in qrels_entries)}")
    print("\nNote: You can manually edit this file to refine relevance judgments.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate qrels.jsonl with real URLs from document retrieval"
    )
    parser.add_argument(
        "--queries",
        type=str,
        default="data/eval/example_queries.txt",
        help="Path to queries file (default: data/eval/example_queries.txt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/eval/qrels.jsonl",
        help="Output qrels file path (default: data/eval/qrels.jsonl)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top documents to use as relevant per query (default: 10)",
    )
    
    args = parser.parse_args()
    
    queries = load_queries(args.queries)
    if not queries:
        print(f"Error: No queries found in {args.queries}")
        return
    
    generate_qrels(queries, args.output, args.top_k)


if __name__ == "__main__":
    main()
