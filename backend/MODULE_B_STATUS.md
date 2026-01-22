# Module B — Query Processing & Cross-Lingual Handling: Completion Status

## ✅ All Requirements Complete

### 1. Language Detection

#### ✅ Implementation
- **Function**: `detect_language_simple()`
- **Returns**: `'bn'`, `'en'`, `'mixed'`, or `'unknown'`
- **Method**: Script-based heuristics (Unicode range detection)
  - Bengali: Unicode range 0x0980-0x09FF
  - English: Latin script (A-Z, a-z)
  - Mixed: Both scripts present
- **Location**: `backend/clir/query_processor.py` (lines 57-75)

**Features**:
- Detects Bengali script characters
- Detects Latin/English script characters
- Handles mixed-language queries
- No heavy dependencies (lightweight implementation)

---

### 2. Normalization

#### ✅ Implementation
- **Function**: `normalize_query()`
- **Features**:
  - ✅ **Lowercase**: Applied to English queries (preserves Bengali case)
  - ✅ **Whitespace Removal**: Removes extra whitespace, normalizes to single spaces
  - ✅ **Punctuation Removal**: Removes punctuation while preserving Bengali characters
  - ✅ **Unicode Normalization**: NFKC normalization for consistent character representation
  - ✅ **Stopword Removal**: Optional via `remove_stopwords()` function
- **Location**: `backend/clir/query_processor.py` (lines 119-135, 144-155)

**Stopword Lists**:
- ✅ English stopwords: 50+ common words (a, an, the, is, am, are, etc.)
- ✅ Bengali stopwords: 30+ common words (এ, এক, এই, সে, etc.)
- **Location**: `backend/clir/query_processor.py` (lines 85-104)

---

### 3. Query Conversion/Translation (Required)

#### ✅ Implementation
- **Class**: `Translator`
- **Backends** (in priority order):
  1. `deep-translator` (GoogleTranslator) - Primary
  2. `googletrans` (fallback) - Secondary
- **Features**:
  - ✅ Translates between Bengali and English
  - ✅ Handles translation errors gracefully
  - ✅ Uses free tools (no paid APIs)
  - ✅ Automatic fallback if one backend fails
- **Location**: `backend/clir/query_processor.py` (lines 202-266)

**Translation Flow**:
- Bengali query → English translation
- English query → Bengali translation
- Mixed query → Translates to target language
- Error handling: Returns `None` if translation fails (doesn't crash)

---

### 4. Query Expansion (Recommended)

#### ✅ Implementation
- **Function**: `expand_query_tokens()`
- **Features**:

#### For Bengali:
- ✅ **Stem Variants**: `generate_bangla_stem_variants()`
  - Removes common Bengali suffixes (গুলো, গুলি, দের, কে, তে, etc.)
  - Example: "শিক্ষা" → ["শিক্ষা", "শিক্ষ"]
- **Location**: `backend/clir/query_processor.py` (lines 279-284)

#### For English:
- ✅ **Basic Variants**: `generate_english_basic_variants()`
  - Handles plurals (adds/removes 's')
  - Handles possessives (removes "'s")
  - Example: "school" → ["school", "schools"]
- ✅ **WordNet Synonyms**: `generate_english_wordnet_synonyms()` (optional)
  - Uses NLTK WordNet for synonyms
  - Example: "education" → ["learning", "instruction", "teaching"]
- **Location**: `backend/clir/query_processor.py` (lines 287-317)

**Expansion Process**:
- Expands primary language tokens
- Expands translated query tokens
- Combines all variants into expanded query set
- **Location**: `backend/clir/query_processor.py` (lines 320-334, 533-554)

---

### 5. Named-Entity Mapping (Recommended)

#### ✅ Named Entity Extraction
- **Function**: `extract_named_entities()`
- **Methods**:
  - **English**: 
    - Uses spaCy NER if available (tries en_core_web_sm/md/lg)
    - Falls back to heuristic pattern matching:
      - Capitalized sequences (e.g., "New York", "Sheikh Hasina")
      - Acronyms (e.g., "USA", "BBC")
  - **Bengali**: 
    - Heuristic: Tokens >= 3 chars, not in stopwords
    - Filters common stopwords
  - **Mixed**: Processes both scripts separately
- **Location**: `backend/clir/query_processor.py` (lines 345-409)

#### ✅ Named Entity Mapping
- **Function**: `map_named_entities()`
- **Default Mappings**: `DEFAULT_NAMED_ENTITY_MAP`
  - "Bangladesh" ↔ "বাংলাদেশ"
  - "Dhaka" ↔ "ঢাকা"
  - "Chattogram" ↔ "চট্টগ্রাম"
  - "Sylhet" ↔ "সিলেট"
  - "Sheikh Hasina" ↔ "শেখ হাসিনা"
  - "Awami League" ↔ "আওয়ামী লীগ"
- **External Mapping**: Supports loading from file via `CLIR_NE_MAP_PATH` environment variable
- **Location**: `backend/clir/query_processor.py` (lines 412-452)

**Mapping Process**:
- Extracts entities from query
- Maps entities using dictionary (bidirectional)
- Adds mapped entities to expanded queries for both languages
- **Location**: `backend/clir/query_processor.py` (lines 558-614)

---

## 📋 Module B Requirements Checklist

### Core Tasks
- ✅ **Language Detection**: Identifies Bangla, English, or Mixed
- ✅ **Normalization**: Lowercase, whitespace removal, optional stopword removal
- ✅ **Translation (Required)**: Translates queries between Bengali and English
- ✅ **Query Expansion (Recommended)**: Synonyms and morphological variants
  - ✅ Bengali: Stem variants (suffix removal)
  - ✅ English: Basic variants + WordNet synonyms
- ✅ **Named-Entity Mapping (Recommended)**: Extracts and maps entities across languages

### Purpose/Goals
- ✅ **Understand translation limitations**: System handles translation errors gracefully
- ✅ **Cross-lingual mismatch handling**: NER mapping addresses proper noun mismatches
- ✅ **Robust error handling**: Translation failures don't crash the system

---

## 🔧 Additional Features

### Enhanced Outputs for Module C
- ✅ **retrieval_queries**: Clean queries for retrieval (no token-soup)
- ✅ **retrieval_keywords**: Important keywords only (stopwords removed)
- **Purpose**: Provides clean input to Module C retrieval models

### Timing Information
- ✅ Tracks timing for each step:
  - Language detection time
  - Normalization time
  - Translation time
  - Expansion time
  - NER and mapping time
  - Total processing time

### Error Handling
- ✅ Translation errors caught and handled
- ✅ Graceful fallback when translation fails
- ✅ Debug information included in results

---

## 🚀 Usage Example

```python
from clir.query_processor import QueryProcessor

processor = QueryProcessor(
    enable_stopwords=True,
    enable_wordnet_expansion=False
)

result = processor.process("ঢাকার খবর")

print(f"Detected Language: {result.detected_language}")
print(f"Normalized: {result.normalized_query}")
print(f"Translated: {result.translated_query}")
print(f"Named Entities: {result.named_entities}")
print(f"Mapped Entities: {result.mapped_entities}")
print(f"BN Retrieval Queries: {result.retrieval_queries['bn']}")
print(f"EN Retrieval Queries: {result.retrieval_queries['en']}")
```

**Output Example**:
```
Detected Language: bn
Normalized: ঢাকার খবর
Translated: news of dhaka
Named Entities: ['ঢাকা']
Mapped Entities: {'ঢাকা': 'Dhaka'}
BN Retrieval Queries: ['ঢাকার খবর']
EN Retrieval Queries: ['news of dhaka', 'Dhaka']
```

---

## ✅ Summary

**Module B is COMPLETE** with all requirements implemented:

1. ✅ **Language Detection**: Script-based detection (Bengali/English/Mixed)
2. ✅ **Normalization**: Lowercase, whitespace, punctuation, optional stopwords
3. ✅ **Translation**: Free tools (deep-translator/googletrans) with error handling
4. ✅ **Query Expansion**: Bengali stems + English variants + WordNet synonyms
5. ✅ **Named-Entity Mapping**: Extraction + bidirectional mapping (Bangladesh ↔ বাংলাদেশ)

**All requirements from the Module B specification are met.**

**Additional Enhancements**:
- Clean retrieval queries for Module C
- Keyword extraction (stopwords removed)
- Comprehensive timing information
- Robust error handling

---

**Last Updated**: 2025-01-XX
