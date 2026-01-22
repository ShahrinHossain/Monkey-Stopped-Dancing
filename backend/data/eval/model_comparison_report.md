# Module C — Model Comparison Report

Generated: 2026-01-23 00:32:22

---

## 1. BM25 vs TF-IDF Comparison

### Analysis

Both BM25 and TF-IDF are lexical retrieval models, but they differ in their ranking strategies:

- **BM25**: Uses term frequency normalization with document length penalization. Better for longer documents and queries with repeated terms.
- **TF-IDF**: Emphasizes rare terms (high IDF). Better for distinguishing documents with unique vocabulary.

### Query: `good and bad news of sylhet`

- **Overlap**: 0/10 documents
- **BM25-only**: 10 documents
- **TF-IDF-only**: 10 documents

**Analysis**: Low overlap (0/10) suggests different ranking strategies. BM25 uses term frequency normalization and document length penalization, while TF-IDF emphasizes rare terms. This difference is expected for queries with varying term frequencies across the corpus. Score difference: BM25 avg=11.963, TF-IDF avg=0.134. BM25 typically performs better for longer queries and documents with varying lengths.

---

### Query: `recent flood in sylhet`

- **Overlap**: 0/10 documents
- **BM25-only**: 10 documents
- **TF-IDF-only**: 10 documents

**Analysis**: Low overlap (0/10) suggests different ranking strategies. BM25 uses term frequency normalization and document length penalization, while TF-IDF emphasizes rare terms. This difference is expected for queries with varying term frequencies across the corpus. Score difference: BM25 avg=10.061, TF-IDF avg=0.353. BM25 typically performs better for longer queries and documents with varying lengths.

---

### Query: `ঢাকার খবর`

- **Overlap**: 0/10 documents
- **BM25-only**: 10 documents
- **TF-IDF-only**: 10 documents

**Analysis**: Low overlap (0/10) suggests different ranking strategies. BM25 uses term frequency normalization and document length penalization, while TF-IDF emphasizes rare terms. This difference is expected for queries with varying term frequencies across the corpus. Score difference: BM25 avg=7.350, TF-IDF avg=0.132. BM25 typically performs better for longer queries and documents with varying lengths.

---

### Query: `বাংলাদেশের অর্থনীতি`

- **Overlap**: 7/10 documents
- **BM25-only**: 3 documents
- **TF-IDF-only**: 3 documents

**Analysis**: Score difference: BM25 avg=7.942, TF-IDF avg=0.310. BM25 typically performs better for longer queries and documents with varying lengths.

---

### Query: `education system in Bangladesh`

- **Overlap**: 0/10 documents
- **BM25-only**: 10 documents
- **TF-IDF-only**: 10 documents

**Analysis**: Low overlap (0/10) suggests different ranking strategies. BM25 uses term frequency normalization and document length penalization, while TF-IDF emphasizes rare terms. This difference is expected for queries with varying term frequencies across the corpus. Score difference: BM25 avg=9.718, TF-IDF avg=0.292. BM25 typically performs better for longer queries and documents with varying lengths.

---

### Query: `শিক্ষা ব্যবস্থা`

- **Overlap**: 0/10 documents
- **BM25-only**: 10 documents
- **TF-IDF-only**: 10 documents

**Analysis**: Low overlap (0/10) suggests different ranking strategies. BM25 uses term frequency normalization and document length penalization, while TF-IDF emphasizes rare terms. This difference is expected for queries with varying term frequencies across the corpus. Score difference: BM25 avg=9.439, TF-IDF avg=0.292. BM25 typically performs better for longer queries and documents with varying lengths.

---

### Query: `news about Dhaka`

- **Overlap**: 0/10 documents
- **BM25-only**: 10 documents
- **TF-IDF-only**: 10 documents

**Analysis**: Low overlap (0/10) suggests different ranking strategies. BM25 uses term frequency normalization and document length penalization, while TF-IDF emphasizes rare terms. This difference is expected for queries with varying term frequencies across the corpus. Score difference: BM25 avg=7.350, TF-IDF avg=0.132. BM25 typically performs better for longer queries and documents with varying lengths.

---

### Query: `সিলেটের বন্যা`

- **Overlap**: 0/10 documents
- **BM25-only**: 10 documents
- **TF-IDF-only**: 10 documents

**Analysis**: Low overlap (0/10) suggests different ranking strategies. BM25 uses term frequency normalization and document length penalization, while TF-IDF emphasizes rare terms. This difference is expected for queries with varying term frequencies across the corpus. Score difference: BM25 avg=10.623, TF-IDF avg=0.319. BM25 typically performs better for longer queries and documents with varying lengths.

---

### Query: `Bangladesh economy`

- **Overlap**: 7/10 documents
- **BM25-only**: 3 documents
- **TF-IDF-only**: 3 documents

**Analysis**: Score difference: BM25 avg=7.941, TF-IDF avg=0.310. BM25 typically performs better for longer queries and documents with varying lengths.

---

### Query: `রাজনীতি`

- **Overlap**: 3/10 documents
- **BM25-only**: 7 documents
- **TF-IDF-only**: 7 documents

**Analysis**: Low overlap (3/10) suggests different ranking strategies. BM25 uses term frequency normalization and document length penalization, while TF-IDF emphasizes rare terms. This difference is expected for queries with varying term frequencies across the corpus. Score difference: BM25 avg=7.088, TF-IDF avg=0.223. BM25 typically performs better for longer queries and documents with varying lengths.

---

## 2. Failure Case Analysis

### Why Lexical Models Fail

#### Synonyms and Paraphrases

Lexical models (BM25, TF-IDF) fail when:
- Documents use synonyms (e.g., 'education' vs 'learning')
- Documents use paraphrases (e.g., 'good news' vs 'positive developments')
- Query and document use different terminology

**Solution**: Use semantic models or query expansion.

#### Cross-Script Terms

Lexical models fail when:
- Query is in Bengali but documents are in English (e.g., 'ঢাকা' vs 'Dhaka')
- Query is in English but documents are in Bengali
- No transliteration or translation mapping exists

**Solution**: Use translation (Module B), transliteration matching (Fuzzy model), or semantic models (multilingual embeddings).

## 3. Model Recommendations

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| Exact keyword matching | BM25 | Best for precise term matching |
| Cross-lingual queries | Semantic | Multilingual embeddings handle translation |
| Mixed queries | Hybrid | Combines lexical + semantic signals |
| Transliteration needed | Fuzzy + Semantic | Fuzzy for edit distance, Semantic for meaning |
| General purpose | Hybrid | Best overall performance |

