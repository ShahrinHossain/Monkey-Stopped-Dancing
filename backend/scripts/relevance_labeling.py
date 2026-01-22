"""
Relevance Labeling Tool for CLIR Evaluation

Manually label query-document pairs as relevant/not relevant.
Outputs CSV format for evaluation.
"""

import argparse
import csv
import json
import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from clir.query_retrieval import QueryRetrievalEngine
from clir.evaluation import RankingAndScoringEngine


def load_queries_from_file(file_path: str) -> List[str]:
    """Load queries from a text file (one per line)."""
    queries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)
    return queries


def load_existing_labels(csv_path: str) -> Dict[str, Dict[str, str]]:
    """Load existing labels from CSV to avoid re-labeling."""
    labels = {}
    if not os.path.exists(csv_path):
        return labels
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['query']}|||{row['doc_url']}"
            labels[key] = row
    return labels


def save_label(csv_path: str, query: str, doc_url: str, language: str, 
               relevant: str, annotator: str) -> None:
    """Append a label to the CSV file."""
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        fieldnames = ["query", "doc_url", "language", "relevant", "annotator"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            "query": query,
            "doc_url": doc_url,
            "language": language,
            "relevant": relevant,
            "annotator": annotator,
        })


def interactive_labeling(queries: List[str], output_csv: str, annotator: str, 
                         top_k: int = 20) -> None:
    """Interactive labeling session."""
    engine = RankingAndScoringEngine()
    existing_labels = load_existing_labels(output_csv)
    
    print(f"\nRelevance Labeling - Annotator: {annotator}")
    print(f"Output: {output_csv} | Queries: {len(queries)}")
    print("Commands: y=relevant, n=not relevant, s=skip, q=quit, next=next query")
    print("-" * 90)
    
    for query_idx, query in enumerate(queries, 1):
        print(f"\n[{query_idx}/{len(queries)}] Query: {query}")
        print("-" * 90)
        
        # Get ranked results
        result = engine.rank(user_query=query, top_k=top_k, model_name="hybrid")
        
        if not result.ranked_documents:
            print("No documents retrieved for this query. Skipping...")
            continue
        
        if result.warning_low_confidence:
            print(f"Warning: Low confidence (score: {result.top_confidence:.2f})")
        
        print(f"\nRetrieved {len(result.ranked_documents)} documents:")
        print("-" * 90)
        
        for rank, doc in enumerate(result.ranked_documents, 1):
            key = f"{query}|||{doc.url}"
            
            # Check if already labeled
            if key in existing_labels:
                existing = existing_labels[key]
                print(f"\n[{rank}] (Already labeled: {existing['relevant']})")
                print(f"Title: {doc.title}")
                print(f"URL: {doc.url}")
                print(f"Language: {doc.language}")
                print(f"Confidence: {doc.matching_confidence:.4f}")
                response = input("Re-label? (y/n/s/q): ").strip().lower()
                if response in ("q", "quit"):
                    print("\nExiting.")
                    return
                if response in ("s", "skip"):
                    continue
                if response not in ("y", "yes"):
                    continue
            else:
                print(f"\n[{rank}] Title: {doc.title}")
                print(f"URL: {doc.url}")
                print(f"Language: {doc.language}")
                print(f"Confidence: {doc.matching_confidence:.4f}")
                if doc.matched_keywords:
                    print(f"Matched keywords: {', '.join(doc.matched_keywords[:5])}")
                if doc.evidence_lines:
                    print(f"Evidence: {doc.evidence_lines[0][:100]}...")
            
            while True:
                response = input("\nRelevant? (y/n/s/q/next): ").strip().lower()
                
                if response in ("q", "quit"):
                    print("\nExiting.")
                    return
                if response in ("s", "skip"):
                    break
                if response in ("next",):
                    # Skip remaining documents for this query
                    break
                if response in ("y", "yes"):
                    save_label(output_csv, query, doc.url, doc.language, "yes", annotator)
                    print("Labeled as RELEVANT")
                    break
                if response in ("n", "no"):
                    save_label(output_csv, query, doc.url, doc.language, "no", annotator)
                    print("Labeled as NOT RELEVANT")
                    break
                print("Invalid input. Please enter y/n/s/q/next")
            
            if response in ("next",):
                break
        
        print("\n" + "=" * 90)
    
    print("\nLabeling complete!")
    print(f"Labels saved to: {output_csv}")


def batch_labeling(queries: List[str], output_csv: str, annotator: str, 
                  top_k: int = 20) -> None:
    """Batch mode: show all queries and documents, then collect labels."""
    engine = RankingAndScoringEngine()
    existing_labels = load_existing_labels(output_csv)
    
    all_docs = []
    
    print("\nBatch mode: Collecting all documents...")
    
    for query in queries:
        result = engine.rank(user_query=query, top_k=top_k, model_name="hybrid")
        for doc in result.ranked_documents:
            key = f"{query}|||{doc.url}"
            if key not in existing_labels:
                all_docs.append((query, doc))
    
    print(f"\nTotal documents to label: {len(all_docs)}")
    print("=" * 90)
    
    for idx, (query, doc) in enumerate(all_docs, 1):
        print(f"\n[{idx}/{len(all_docs)}]")
        print(f"Query: {query}")
        print(f"Title: {doc.title}")
        print(f"URL: {doc.url}")
        print(f"Language: {doc.language}")
        print(f"Confidence: {doc.matching_confidence:.4f}")
        
        while True:
            response = input("Relevant? (y/n/s/q): ").strip().lower()
            if response in ("q", "quit"):
                    print("\nExiting.")
                return
            if response in ("s", "skip"):
                break
            if response in ("y", "yes"):
                save_label(output_csv, query, doc.url, doc.language, "yes", annotator)
                print("Labeled as RELEVANT")
                break
            if response in ("n", "no"):
                save_label(output_csv, query, doc.url, doc.language, "no", annotator)
                print("Labeled as NOT RELEVANT")
                break
            print("Invalid input. Please enter y/n/s/q")
    
    print("\nLabeling complete!")
    print(f"Labels saved to: {output_csv}")


def convert_to_qrels(csv_path: str, qrels_path: str) -> None:
    """Convert labeled CSV to QRELS JSONL format."""
    query_to_urls: Dict[str, List[str]] = {}
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row["query"]
            doc_url = row["doc_url"]
            relevant = row["relevant"].lower()
            
            if relevant in ("yes", "y", "1", "true"):
                if query not in query_to_urls:
                    query_to_urls[query] = []
                query_to_urls[query].append(doc_url)
    
    # Write QRELS JSONL
    with open(qrels_path, "w", encoding="utf-8") as f:
        for query, urls in query_to_urls.items():
            qrel_entry = {
                "query": query,
                "relevant_urls": urls
            }
            f.write(json.dumps(qrel_entry, ensure_ascii=False) + "\n")
    
    print(f"Converted {len(query_to_urls)} queries to QRELS format: {qrels_path}")


def main():
    parser = argparse.ArgumentParser(description="Relevance Labeling Tool for CLIR")
    parser.add_argument(
        "--queries",
        type=str,
        help="Path to text file with queries (one per line)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/eval/labels.csv",
        help="Output CSV file path (default: data/eval/labels.csv)",
    )
    parser.add_argument(
        "--annotator",
        type=str,
        default="annotator1",
        help="Annotator name (default: annotator1)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top documents to show per query (default: 20)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use batch mode (collect all docs first, then label)",
    )
    parser.add_argument(
        "--convert-to-qrels",
        type=str,
        help="Convert labeled CSV to QRELS JSONL format (provide output path)",
    )
    
    args = parser.parse_args()
    
    # Convert to QRELS mode
    if args.convert_to_qrels:
        if not os.path.exists(args.output):
            print(f"Error: Labels file not found: {args.output}")
            return
        convert_to_qrels(args.output, args.convert_to_qrels)
        return
    
    # Load queries
    if args.queries:
        queries = load_queries_from_file(args.queries)
    else:
        # Default queries for testing
        queries = [
            "good and bad news of sylhet",
            "recent flood in sylhet",
            "ঢাকার খবর",
            "বাংলাদেশের অর্থনীতি",
        ]
        print("No --queries file provided. Using default queries.")
        print("You can create a queries.txt file with one query per line.")
    
    if not queries:
        print("Error: No queries to label.")
        return
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Run labeling
    if args.batch:
        batch_labeling(queries, args.output, args.annotator, args.top_k)
    else:
        interactive_labeling(queries, args.output, args.annotator, args.top_k)


if __name__ == "__main__":
    main()
