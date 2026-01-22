"""
Verify Evaluation Metrics Against Target Thresholds

Checks if metrics meet targets: Precision@10 >= 0.6, Recall@50 >= 0.5,
nDCG@10 >= 0.5, MRR >= 0.4
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any  

sys.path.insert(0, str(Path(__file__).parent.parent))

from clir.evaluation import Evaluator, load_qrels_jsonl, print_summary


# Target thresholds
TARGETS = {
    "Precision@10": 0.6,
    "Recall@50": 0.5,
    "nDCG@10": 0.5,
    "MRR": 0.4,
}


def verify_metrics(summary: dict) -> Tuple[bool, List[str]]:
    passed = True
    issues = []
    
    for metric_name, target_value in TARGETS.items():
        actual_value = summary.get(metric_name, 0.0)
        
        if actual_value < target_value:
            passed = False
            issues.append(
                f"FAIL {metric_name}: {actual_value:.3f} < {target_value:.3f}"
            )
        else:
            issues.append(
                f"PASS {metric_name}: {actual_value:.3f} >= {target_value:.3f}"
            )
    
    return passed, issues


def main():
    parser = argparse.ArgumentParser(
        description="Verify CLIR evaluation metrics against target thresholds"
    )
    parser.add_argument(
        "--qrels",
        type=str,
        required=True,
        help="Path to QRELS JSONL file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hybrid",
        choices=["bm25", "tfidf", "fuzzy", "semantic", "hybrid"],
        help="Retrieval model to evaluate",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-K documents for ranking (default: 10)",
    )
    
    args = parser.parse_args()
    
    print(f"\nCLIR Metrics Verification - Model: {args.model}")
    print(f"QRELS: {args.qrels}")
    print("Target Thresholds:")
    for metric, target in TARGETS.items():
        print(f"  {metric}: >= {target:.2f}")
    print()
    
    try:
        qrels_items = load_qrels_jsonl(args.qrels)
        print(f"Loaded {len(qrels_items)} queries from QRELS file\n")
    except Exception as e:
        print(f"Error loading QRELS: {e}")
        return
    
    evaluator = Evaluator(
        model_name=args.model,
        top_k_for_ranking=args.top_k,
    )
    
    all_results, summary = evaluator.evaluate_queries(qrels_items)
    
    # Print summary
    print_summary(summary)
    
    passed, issues = verify_metrics(summary)
    
    print("\nVerification Results:")
    for issue in issues:
        print(issue)
    print()
    if passed:
        print("ALL TARGETS MET")
        return 0
    else:
        print("SOME TARGETS NOT MET")
        print("\nRecommendations:")
        print("  1. Review query processing and translation quality")
        print("  2. Improve retrieval models (especially semantic matching)")
        print("  3. Tune hybrid model weights")
        print("  4. Expand and improve the dataset")
        print("  5. Review QRELS labels for accuracy")
        return 1


if __name__ == "__main__":
    sys.exit(main())
