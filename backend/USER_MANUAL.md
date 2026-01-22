# CLIR System - User Manual

Complete guide on how to run each module and tool in the Cross-Language Information Retrieval (CLIR) system.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Module B - Query Processing](#module-b---query-processing)
3. [Module C - Retrieval Models](#module-c---retrieval-models)
4. [Module D - Ranking & Evaluation](#module-d---ranking--evaluation)
5. [Utility Tools](#utility-tools)
6. [Complete Workflow Examples](#complete-workflow-examples)

---

## Prerequisites

### Installation

```bash
cd backend
pip install -r requirements.txt
```

### Required Data Files

Ensure you have:
- `data/processed/bn.jsonl` - Bengali documents
- `data/processed/en.jsonl` - English documents
- `data/eval/qrels.jsonl` - Ground truth for evaluation (optional)

---

## Module B - Query Processing

### Purpose
Processes user queries: detects language, normalizes, translates, expands, and extracts named entities.

### Basic Usage

#### Interactive Mode (CLI)
```bash
cd backend
python -m clir.query_processor
```

**Example Session**:
```
Query (empty to quit): ঢাকার খবর
[Output shows: detected language, normalized query, translated query, named entities, etc.]
```

#### Programmatic Usage
```python
from clir.query_processor import QueryProcessor

# Initialize processor
processor = QueryProcessor(
    enable_stopwords=True,        # Enable stopword removal
    enable_wordnet_expansion=False # Disable WordNet (faster)
)

# Process a query
result = processor.process("ঢাকার খবর")

# Access results
print(f"Detected Language: {result.detected_language}")
print(f"Normalized: {result.normalized_query}")
print(f"Translated: {result.translated_query}")
print(f"Named Entities: {result.named_entities}")
print(f"Mapped Entities: {result.mapped_entities}")
print(f"BN Retrieval Queries: {result.retrieval_queries['bn']}")
print(f"EN Retrieval Queries: {result.retrieval_queries['en']}")
print(f"BN Keywords: {result.retrieval_keywords['bn']}")
print(f"EN Keywords: {result.retrieval_keywords['en']}")
```

### Configuration Options

```python
processor = QueryProcessor(
    enable_stopwords=True,           # Remove stopwords from queries
    enable_wordnet_expansion=True,    # Use WordNet for English synonyms (slower)
    translator=None,                 # Custom translator (default: Google Translate)
    named_entity_map=None            # Custom NE map (default: built-in + env var)
)
```

### Output Fields

- `original_query`: Original user input
- `detected_language`: 'bn', 'en', 'mixed', or 'unknown'
- `normalized_query`: Cleaned and normalized query
- `translated_query`: Translated version (if translation succeeded)
- `expanded_queries`: Dictionary with 'bn' and 'en' expanded query lists
- `named_entities`: List of extracted named entities
- `mapped_entities`: Dictionary mapping entities across languages
- `retrieval_queries`: Clean queries for Module C (no token-soup)
- `retrieval_keywords`: Important keywords only (stopwords removed)
- `timings_ms`: Timing breakdown for each step

---

## Module C - Retrieval Models

### Purpose
Retrieves relevant documents using multiple models: BM25, TF-IDF, Fuzzy, Semantic, and Hybrid.

### Basic Usage

#### Interactive Mode (CLI)
```bash
cd backend
python -m clir.query_retrieval
```

**Example Session**:
```
Query (empty to quit): good and bad news of sylhet
[Shows results from all models: BM25, TF-IDF, Fuzzy, Semantic, Hybrid]
```

#### Programmatic Usage

```python
from clir.query_retrieval import QueryRetrievalEngine

# Initialize engine (loads documents and builds indexes)
engine = QueryRetrievalEngine(
    bangla_jsonl_path="data/processed/bn.jsonl",
    english_jsonl_path="data/processed/en.jsonl",
    embedding_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    enable_stopwords=True
)

# Search with a specific model
results = engine.search(
    user_query="ঢাকার খবর",
    top_k=10,
    model="hybrid",  # Options: "bm25", "tfidf", "fuzzy", "semantic", "hybrid", "all"
    include_debug=True
)

# Access results
print("Bengali Results (Hybrid):")
for doc in results["bn"]["hybrid"]:
    print(f"  - {doc['title']} (score: {doc['score']:.4f})")
    print(f"    URL: {doc['url']}")
    print(f"    Keywords: {', '.join(doc['matched_keywords'])}")

print("\nEnglish Results (Hybrid):")
for doc in results["en"]["hybrid"]:
    print(f"  - {doc['title']} (score: {doc['score']:.4f})")
```

### Available Models

1. **`"bm25"`**: Okapi BM25 lexical retrieval
2. **`"tfidf"`**: TF-IDF lexical retrieval
3. **`"fuzzy"`**: Fuzzy/transliteration matching (RapidFuzz)
4. **`"semantic"`**: Semantic matching (multilingual embeddings)
5. **`"hybrid"`**: Weighted fusion of all models (default weights)
6. **`"all"`**: Returns results from all models

### Custom Hybrid Weights

```python
results = engine.search(
    user_query="education in Bangladesh",
    top_k=10,
    model="hybrid",
    hybrid_weights={
        "bm25": 0.30,
        "tfidf": 0.10,
        "fuzzy": 0.20,
        "semantic": 0.40
    }
)
```

### First-Time Setup

**Note**: On first run, the semantic model will:
1. Download the embedding model (~400MB)
2. Compute embeddings for all documents (takes time)
3. Cache embeddings to `models/embeddings_cache/` for future use

**Time Estimates**:
- Small dataset (< 1000 docs): ~5-10 minutes
- Medium dataset (1000-5000 docs): ~15-30 minutes
- Large dataset (> 5000 docs): ~30-60 minutes

---

## Module D - Ranking & Evaluation

### Purpose
Ranks documents, assigns confidence scores, and evaluates system performance.

### Basic Usage

#### Demo Query (No QRELS Required)
```bash
cd backend
python -m clir.evaluation --demo_query "good and bad news of sylhet" --model hybrid --top_k 10
```

**Output**:
- Ranked documents with confidence scores
- Low-confidence warning (if applicable)
- Timing breakdown

#### Full Evaluation (Requires QRELS)
```bash
python -m clir.evaluation \
    --qrels data/eval/qrels.jsonl \
    --model hybrid \
    --top_k 10 \
    --low_conf 0.20
```

**Output**:
- Per-query ranked results
- Per-query metrics (Precision@10, Recall@50, nDCG@10, MRR)
- Overall summary metrics

### Command-Line Options

```bash
python -m clir.evaluation \
    --qrels <path>          # Path to QRELS JSONL file
    --model <model>         # Model: bm25, tfidf, fuzzy, semantic, hybrid
    --top_k <number>        # Top-K documents to show (default: 10)
    --low_conf <threshold>  # Low-confidence threshold (default: 0.20)
    --demo_query <text>     # Run single query demo (no QRELS needed)
```

### Programmatic Usage

```python
from clir.evaluation import RankingAndScoringEngine, Evaluator

# For single query ranking
ranking_engine = RankingAndScoringEngine()
result = ranking_engine.rank(
    user_query="ঢাকার খবর",
    model_name="hybrid",
    top_k=10
)

print(f"Top Confidence: {result.top_confidence:.4f}")
print(f"Warning: {result.warning_low_confidence}")
for doc in result.ranked_documents:
    print(f"  {doc.title} (confidence: {doc.matching_confidence:.4f})")

# For full evaluation
evaluator = Evaluator(
    model_name="hybrid",
    top_k_for_ranking=10,
    low_confidence_threshold=0.20
)

qrels_items = [
    {
        "query": "good and bad news of sylhet",
        "relevant_urls": ["https://example.com/news1", "https://example.com/news2"]
    }
]

all_results, summary = evaluator.evaluate_queries(qrels_items)

print(f"Precision@10: {summary['Precision@10']:.3f}")
print(f"Recall@50: {summary['Recall@50']:.3f}")
print(f"nDCG@10: {summary['nDCG@10']:.3f}")
print(f"MRR: {summary['MRR']:.3f}")
```

### QRELS File Format

Create `data/eval/qrels.jsonl` with one JSON object per line:

```jsonl
{"query": "good and bad news of sylhet", "relevant_urls": ["https://example.com/news1", "https://example.com/news2"]}
{"query": "ঢাকার খবর", "relevant_urls": ["https://example.com/dhaka1"]}
```

**Quick Generation**: To automatically generate a qrels file from query results:

```bash
cd backend
python -m scripts.generate_qrels \
    --queries data/eval/example_queries.txt \
    --output data/eval/qrels.jsonl \
    --top_k 10
```

This runs each query, retrieves top-k documents, and uses their URLs as "relevant" in the qrels file. Useful for testing, but metrics may be artificially high. For accurate evaluation, use the relevance labeling tool below.

---

## Utility Tools

### 1. Generate QRELS from Query Results

**Purpose**: Automatically generate qrels.jsonl by running queries and using retrieved documents as relevant.

```bash
cd backend
python -m scripts.generate_qrels \
    --queries data/eval/example_queries.txt \
    --output data/eval/qrels.jsonl \
    --top_k 10
```

**Options**:
- `--queries`: Path to queries file (default: `data/eval/example_queries.txt`)
- `--output`: Output qrels file path (default: `data/eval/qrels.jsonl`)
- `--top_k`: Number of top documents to use as relevant per query (default: 10)

**Note**: This method marks retrieved documents as relevant, so evaluation metrics will be high. For accurate evaluation, use the relevance labeling tool below.

### 2. Relevance Labeling Tool

**Purpose**: Manually label query-document pairs as relevant/not relevant.

#### Interactive Mode
```bash
cd backend
python -m scripts.relevance_labeling \
    --queries data/eval/example_queries.txt \
    --output data/eval/labels.csv \
    --annotator your_name \
    --top_k 20
```

**During Labeling**:
- Shows top-K documents for each query
- Type `y` or `yes` for relevant
- Type `n` or `no` for not relevant
- Type `s` or `skip` to skip a document
- Type `next` to move to next query
- Type `q` or `quit` to exit

#### Batch Mode
```bash
python -m scripts.relevance_labeling \
    --queries data/eval/example_queries.txt \
    --output data/eval/labels.csv \
    --annotator your_name \
    --batch
```

#### Convert to QRELS Format
```bash
python -m scripts.relevance_labeling \
    --convert-to-qrels data/eval/qrels.jsonl
```

**Input**: `data/eval/labels.csv` (must exist)
**Output**: `data/eval/qrels.jsonl`

### 2. Error Analysis Tool

**Purpose**: Analyzes retrieval failures across 5 categories.

```bash
cd backend
python -m scripts.error_analysis \
    --queries data/eval/example_queries.txt \
    --output data/eval/error_analysis_report.md
```

**Output**: Markdown report with:
- Translation failures
- Named Entity mismatches
- Semantic vs. Lexical comparisons
- Cross-script ambiguity
- Code-switching analysis

**Example Queries File** (`data/eval/example_queries.txt`):
```
good and bad news of sylhet
recent flood in sylhet
ঢাকার খবর
বাংলাদেশের অর্থনীতি
education system in Bangladesh
```

### 3. Model Comparison Tool

**Purpose**: Compares different retrieval models and analyzes failure cases.

```bash
cd backend
python -m scripts.model_comparison \
    --queries data/eval/example_queries.txt \
    --output data/eval/model_comparison_report.md \
    --analysis all
```

**Analysis Types**:
- `all`: All analyses
- `bm25-tfidf`: Compare BM25 vs TF-IDF
- `failures`: Analyze failure cases
- `all-models`: Compare all models

### 4. Metrics Verification Tool

**Purpose**: Verifies if evaluation metrics meet target thresholds.

```bash
cd backend
python -m scripts.verify_metrics \
    --qrels data/eval/qrels.jsonl \
    --model hybrid \
    --top_k 10
```

**Target Thresholds**:
- Precision@10 >= 0.6
- Recall@50 >= 0.5
- nDCG@10 >= 0.5
- MRR >= 0.4

**Output**: Pass/fail status with recommendations if targets not met.

---

## Complete Workflow Examples

### Workflow 1: End-to-End Search

```bash
# 1. Process a query (Module B)
python -m clir.query_processor
# Enter: "ঢাকার খবর"

# 2. Retrieve documents (Module C)
python -m clir.query_retrieval
# Enter: "ঢাকার খবর"

# 3. Rank and evaluate (Module D)
python -m clir.evaluation --demo_query "ঢাকার খবর" --model hybrid
```

### Workflow 2: Full Evaluation Pipeline

```bash
# 1. Label queries (if not already done)
python -m scripts.relevance_labeling \
    --queries data/eval/example_queries.txt \
    --output data/eval/labels.csv \
    --annotator annotator1

# 2. Convert labels to QRELS
python -m scripts.relevance_labeling \
    --convert-to-qrels data/eval/qrels.jsonl

# 3. Run evaluation
python -m clir.evaluation \
    --qrels data/eval/qrels.jsonl \
    --model hybrid \
    --top_k 10

# 4. Verify metrics
python -m scripts.verify_metrics \
    --qrels data/eval/qrels.jsonl \
    --model hybrid

# 5. Analyze errors
python -m scripts.error_analysis \
    --queries data/eval/example_queries.txt \
    --output data/eval/error_analysis_report.md

# 6. Compare models
python -m scripts.model_comparison \
    --queries data/eval/example_queries.txt \
    --output data/eval/model_comparison_report.md
```

### Workflow 3: Model Development & Testing

```bash
# 1. Test individual models
python -m clir.evaluation --demo_query "test query" --model bm25
python -m clir.evaluation --demo_query "test query" --model semantic
python -m clir.evaluation --demo_query "test query" --model hybrid

# 2. Compare models
python -m scripts.model_comparison \
    --queries data/eval/test_queries.txt \
    --output comparison.md \
    --analysis all-models

# 3. Analyze failures
python -m scripts.error_analysis \
    --queries data/eval/test_queries.txt \
    --output failures.md
```

---

## Troubleshooting

### Common Issues

#### 1. Translation Errors
**Problem**: `TranslationError: No Google translation backend available`

**Solution**:
```bash
pip install deep-translator googletrans==4.0.0rc1
```

#### 2. Missing Embeddings
**Problem**: Semantic model fails or is slow on first run

**Solution**: 
- First run will download model and compute embeddings
- Embeddings are cached in `models/embeddings_cache/`
- Subsequent runs are fast

#### 3. Missing Data Files
**Problem**: `FileNotFoundError: Dataset file not found`

**Solution**:
- Ensure `data/processed/bn.jsonl` and `data/processed/en.jsonl` exist
- Run the crawler: `python -m scripts.build_dataset`

#### 4. spaCy NER Not Working
**Problem**: NER extraction fails for English

**Solution**:
- Install spaCy model: `python -m spacy download en_core_web_sm`
- Or rely on heuristic fallback (works without spaCy)

#### 5. WordNet Not Available
**Problem**: WordNet expansion fails

**Solution**:
```python
# Disable WordNet (uses basic expansion only)
processor = QueryProcessor(enable_wordnet_expansion=False)
```

Or install NLTK data:
```python
import nltk
nltk.download('wordnet')
```

---

## Performance Tips

### Speed Optimization

1. **Disable WordNet**: Set `enable_wordnet_expansion=False` (faster query processing)
2. **Use Cached Embeddings**: First run computes embeddings, subsequent runs use cache
3. **Limit Top-K**: Use smaller `top_k` values for faster retrieval
4. **Skip Debug Info**: Set `include_debug=False` in Module C

### Memory Optimization

1. **Process in Batches**: For large datasets, process queries in batches
2. **Clear Cache**: Delete `models/embeddings_cache/` if running out of disk space
3. **Use Smaller Models**: Consider smaller embedding models for limited memory

---

## Quick Reference

### Module B (Query Processing)
```bash
python -m clir.query_processor
```

### Module C (Retrieval)
```bash
python -m clir.query_retrieval
```

### Module D (Evaluation)
```bash
# Demo
python -m clir.evaluation --demo_query "your query" --model hybrid

# Full evaluation
python -m clir.evaluation --qrels data/eval/qrels.jsonl --model hybrid
```

### Tools
```bash
# Label queries
python -m scripts.relevance_labeling --queries queries.txt --output labels.csv

# Error analysis
python -m scripts.error_analysis --queries queries.txt --output report.md

# Model comparison
python -m scripts.model_comparison --queries queries.txt --output comparison.md

# Verify metrics
python -m scripts.verify_metrics --qrels qrels.jsonl --model hybrid
```

---

## File Structure

```
backend/
├── clir/
│   ├── query_processor.py    # Module B
│   ├── query_retrieval.py    # Module C
│   └── evaluation.py          # Module D
├── scripts/
│   ├── relevance_labeling.py
│   ├── error_analysis.py
│   ├── model_comparison.py
│   └── verify_metrics.py
├── data/
│   ├── processed/
│   │   ├── bn.jsonl
│   │   └── en.jsonl
│   └── eval/
│       ├── qrels.jsonl
│       ├── labels.csv
│       └── example_queries.txt
└── models/
    └── embeddings_cache/
```

---

**Last Updated**: 2025-01-XX
