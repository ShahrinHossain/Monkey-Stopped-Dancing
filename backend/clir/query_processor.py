# backend/clir/query_processor.py
"""
Module B — Query Processing & Cross-Lingual Handling

Implements language detection, normalization, translation, query expansion,
and named entity extraction/mapping for cross-lingual information retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import os
import re
import json
import time
import unicodedata


# -----------------------------
# Utilities: script detection
# -----------------------------

BENGALI_UNICODE_RANGE = (0x0980, 0x09FF)


def is_bengali_character(character: str) -> bool:
    if not character:
        return False
    unicode_code_point = ord(character)
    return BENGALI_UNICODE_RANGE[0] <= unicode_code_point <= BENGALI_UNICODE_RANGE[1]


def contains_bengali_script(text: str) -> bool:
    return any(is_bengali_character(character) for character in (text or ""))


def contains_latin_script(text: str) -> bool:
    return any(("A" <= character <= "Z") or ("a" <= character <= "z") for character in (text or ""))


def detect_language_simple(raw_query: str) -> str:
    """
    Returns: 'bn', 'en', 'mixed', 'unknown'
    Uses script heuristics to avoid heavy dependencies.
    """
    raw_query = (raw_query or "").strip()
    if not raw_query:
        return "unknown"

    has_bengali = contains_bengali_script(raw_query)
    has_latin = contains_latin_script(raw_query)

    if has_bengali and has_latin:
        return "mixed"
    if has_bengali:
        return "bn"
    if has_latin:
        return "en"
    return "unknown"


# -----------------------------
# Stopwords
# -----------------------------

# You said you've updated EN_STOPWORDS already—keep your version here.
# This file includes a strong default set anyway.

EN_STOPWORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "me", "my", "your", "our", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom", "where", "when", "why", "how",
    "to", "of", "in", "on", "at", "for", "from", "with", "as", "by", "and", "or", "but", "if", "then",
    "about", "into", "over", "under", "between", "within", "without",
    "do", "does", "did", "done", "tell", "say", "give", "show", "explain",
    "latest", "news", "today", "now", "please", "there", "here", "have", "has", "had",
    "will", "would", "should", "can", "could", "may", "might", "must",
}

BN_STOPWORDS = {
    "এ", "এক", "একটি", "এই", "সেই", "ওই", "তা", "এটা", "ওটা", "যে", "যা", "যদি", "তবে",
    "কি", "কী", "কেন", "কবে", "কখন", "কোথায়", "কোথা", "কিভাবে", "কেমন",
    "আমি", "তুমি", "সে", "আমরা", "তারা", "আমার", "তোমার", "তার", "আমাদের", "তাদের",
    "হয়", "হচ্ছে", "ছিল", "ছিলো", "হবে", "হতে",
    "এর", "এবং", "ও", "বা", "কিন্তু", "কারণ",
    "কে", "কাকে", "কার", "দিকে", "জন্য", "থেকে", "উপর", "নিচে", "মধ্যে", "সাথে",
    "বল", "বলো", "বলুন", "দাও", "দিন", "দেখাও", "ব্যাখ্যা",
}

BANGLA_STOPWORDS_MINI = BN_STOPWORDS


# -----------------------------
# Normalization
# -----------------------------

PUNCTUATION_REMOVAL_PATTERN = re.compile(r"[^\w\s\u0980-\u09FF]+", flags=re.UNICODE)
WHITESPACE_NORMALIZATION_PATTERN = re.compile(r"\s+", flags=re.UNICODE)
TOKEN_CLEAN_PATTERN = re.compile(r"[^\w\u0980-\u09FF]+", flags=re.UNICODE)


def normalize_query(raw_query: str, language_code: str) -> str:
    """
    - Trim
    - Unicode normalize (NFKC)
    - Remove odd punctuation (keep Bengali letters)
    - Collapse whitespace
    - Lowercase only for English
    """
    normalized_query_text = (raw_query or "").strip()
    normalized_query_text = unicodedata.normalize("NFKC", normalized_query_text)
    normalized_query_text = PUNCTUATION_REMOVAL_PATTERN.sub(" ", normalized_query_text)
    normalized_query_text = WHITESPACE_NORMALIZATION_PATTERN.sub(" ", normalized_query_text).strip()

    if language_code == "en":
        normalized_query_text = normalized_query_text.lower()

    return normalized_query_text


def tokenize_simple(text: str) -> List[str]:
    if not text:
        return []
    return [token for token in text.split(" ") if token]


def remove_stopwords(token_list: List[str], language_code: str, enabled: bool) -> List[str]:
    if not enabled:
        return token_list

    if language_code == "en":
        stopword_set = EN_STOPWORDS
    elif language_code == "bn":
        stopword_set = BN_STOPWORDS
    else:
        return token_list

    return [token for token in token_list if token and token not in stopword_set]


def _clean_token(token: str, language_code: str) -> str:
    token = (token or "").strip()
    if language_code == "en":
        token = token.lower()
    token = TOKEN_CLEAN_PATTERN.sub("", token)
    return token


def extract_keywords_for_retrieval(text: str, language_code: str, min_len: int) -> List[str]:
    """
    Extracts informative keywords from a text to use for:
    - evidence matching
    - optional boosting

    Always removes stopwords here (hard ON) so you don't accidentally
    match words like: what/are/the/and
    """
    normalized = normalize_query(text or "", language_code)
    tokens = [_clean_token(t, language_code) for t in tokenize_simple(normalized)]
    tokens = [t for t in tokens if t and len(t) >= min_len]

    # hard-remove stopwords for keywords
    if language_code == "en":
        tokens = [t for t in tokens if t not in EN_STOPWORDS]
    elif language_code == "bn":
        tokens = [t for t in tokens if t not in BN_STOPWORDS]

    seen = set()
    ordered: List[str] = []
    for t in tokens:
        if t not in seen:
            ordered.append(t)
            seen.add(t)
    return ordered


# -----------------------------
# Translation (Required)
# -----------------------------

class TranslationError(RuntimeError):
    pass


class Translator:
    """
    Google-Translate based translation (no Transformers, no Keras).
    Priority:
      1) deep-translator (GoogleTranslator)
      2) googletrans (fallback)
    """

    def __init__(self) -> None:
        self._backend_name = None
        self._backend = None
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            from deep_translator import GoogleTranslator  # type: ignore
            self._backend_name = "deep_translator"
            self._backend = GoogleTranslator
            return
        except Exception:
            pass

        try:
            from googletrans import Translator as GoogleTransTranslator  # type: ignore
            self._backend_name = "googletrans"
            self._backend = GoogleTransTranslator()
            return
        except Exception:
            pass

        self._backend_name = None
        self._backend = None

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        input_text = (text or "").strip()
        if not input_text:
            return ""

        if source_language == target_language:
            return input_text

        if self._backend is None or self._backend_name is None:
            raise TranslationError(
                "No Google translation backend available. Install:\n"
                "pip install deep-translator googletrans==4.0.0rc1"
            )

        src = source_language
        dest = target_language

        if self._backend_name == "deep_translator":
            try:
                translator = self._backend(source=src, target=dest)  # type: ignore
                return translator.translate(input_text)
            except Exception as e:
                raise TranslationError(f"deep-translator translation failed: {e}") from e

        if self._backend_name == "googletrans":
            try:
                result = self._backend.translate(input_text, src=src, dest=dest)  # type: ignore
                return result.text
            except Exception as e:
                raise TranslationError(f"googletrans translation failed: {e}") from e

        raise TranslationError("Unknown translation backend configuration.")


# -----------------------------
# Query Expansion (Recommended)
# -----------------------------

BANGLA_SUFFIXES = [
    "গুলো", "গুলি", "দের", "গুলোকে", "গুলোয়", "গুলোতে", "গুলোয়",
    "কে", "তে", "টা", "টি", "টার", "টির", "য়", "ের", "রা", "গুলা"
]


def generate_bangla_stem_variants(token: str) -> List[str]:
    candidate_variants = {token}
    for suffix in BANGLA_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            candidate_variants.add(token[: -len(suffix)])
    return sorted(candidate_variants)


def generate_english_basic_variants(token: str) -> List[str]:
    candidate_variants = {token}
    if token.endswith("'s"):
        candidate_variants.add(token[:-2])
    if token.endswith("s") and len(token) > 3:
        candidate_variants.add(token[:-1])
    else:
        if len(token) > 2:
            candidate_variants.add(token + "s")
    return sorted(candidate_variants)


def generate_english_wordnet_synonyms(token: str, limit: int = 3) -> List[str]:
    try:
        from nltk.corpus import wordnet as wordnet  # type: ignore
    except Exception:
        return []

    synonym_set = set()
    try:
        for synset in wordnet.synsets(token):
            for lemma in synset.lemma_names():
                lemma = lemma.replace("_", " ").lower()
                if lemma != token:
                    synonym_set.add(lemma)
            if len(synonym_set) >= limit:
                break
    except Exception:
        return []

    return sorted(list(synonym_set))[:limit]


def expand_query_tokens(token_list: List[str], language_code: str, use_wordnet: bool = False) -> List[str]:
    expanded_token_set = set(token_list)

    for token in token_list:
        if language_code == "bn":
            for variant in generate_bangla_stem_variants(token):
                expanded_token_set.add(variant)
        elif language_code == "en":
            for variant in generate_english_basic_variants(token):
                expanded_token_set.add(variant)
            if use_wordnet:
                for synonym in generate_english_wordnet_synonyms(token):
                    expanded_token_set.add(synonym)

    return sorted([token for token in expanded_token_set if token])


# -----------------------------
# Named Entity Extraction + Mapping (Recommended)
# -----------------------------

CAPITALIZED_SEQUENCE_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
ACRONYM_PATTERN = re.compile(r"\b([A-Z]{2,})\b")


def extract_named_entities(raw_query: str, language_code: str) -> List[str]:
    """
    Heuristic NER fallback.
    """
    query_text = (raw_query or "").strip()
    if not query_text:
        return []

    if language_code == "en":
        try:
            import spacy  # type: ignore
            for spacy_model_name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
                try:
                    nlp_pipeline = spacy.load(spacy_model_name)
                    spacy_document = nlp_pipeline(query_text)
                    extracted_entities = [entity.text.strip() for entity in spacy_document.ents if entity.text.strip()]
                    if extracted_entities:
                        return sorted(list(dict.fromkeys(extracted_entities)))
                    break
                except Exception:
                    continue
        except Exception:
            pass

        entity_set = set()
        for match in CAPITALIZED_SEQUENCE_PATTERN.finditer(query_text):
            entity_set.add(match.group(1).strip())
        for match in ACRONYM_PATTERN.finditer(query_text):
            entity_set.add(match.group(1).strip())
        return sorted(list(entity_set))

    if language_code == "bn":
        normalized_bangla = normalize_query(query_text, "bn")
        token_list = tokenize_simple(normalized_bangla)
        entity_candidates = [token for token in token_list if len(token) >= 3 and token not in BANGLA_STOPWORDS_MINI]

        seen_entities = set()
        ordered_entities: List[str] = []
        for entity in entity_candidates:
            if entity not in seen_entities:
                ordered_entities.append(entity)
                seen_entities.add(entity)
        return ordered_entities

    if language_code == "mixed":
        token_list = tokenize_simple(query_text)
        bengali_tokens = [token for token in token_list if contains_bengali_script(token)]
        latin_tokens = [token for token in token_list if contains_latin_script(token)]

        bengali_part = " ".join(bengali_tokens)
        latin_part = " ".join(latin_tokens)

        mixed_entities: List[str] = []
        mixed_entities.extend(extract_named_entities(latin_part, "en"))
        mixed_entities.extend(extract_named_entities(bengali_part, "bn"))

        seen_entities = set()
        ordered_entities = []
        for entity in mixed_entities:
            if entity and entity not in seen_entities:
                ordered_entities.append(entity)
                seen_entities.add(entity)
        return ordered_entities

    return []


DEFAULT_NAMED_ENTITY_MAP: Dict[str, str] = {
    "Bangladesh": "বাংলাদেশ",
    "বাংলাদেশ": "Bangladesh",
    "Dhaka": "ঢাকা",
    "ঢাকা": "Dhaka",
    "Chattogram": "চট্টগ্রাম",
    "চট্টগ্রাম": "Chattogram",
    "Sylhet": "সিলেট",
    "সিলেট": "Sylhet",
    "Sheikh Hasina": "শেখ হাসিনা",
    "শেখ হাসিনা": "Sheikh Hasina",
    "Awami League": "আওয়ামী লীগ",
    "আওয়ামী লীগ": "Awami League",
}


def load_external_named_entity_map() -> Dict[str, str]:
    external_map_path = os.getenv("CLIR_NE_MAP_PATH", "").strip()
    if not external_map_path:
        return {}
    try:
        with open(external_map_path, "r", encoding="utf-8") as file_handle:
            map_data = json.load(file_handle)
        if isinstance(map_data, dict):
            return {str(key): str(value) for key, value in map_data.items()}
    except Exception:
        return {}
    return {}


def build_named_entity_mapping() -> Dict[str, str]:
    named_entity_map = dict(DEFAULT_NAMED_ENTITY_MAP)
    named_entity_map.update(load_external_named_entity_map())
    return named_entity_map


def map_named_entities(entity_list: List[str], named_entity_map: Dict[str, str]) -> Dict[str, str]:
    mapped_entities: Dict[str, str] = {}
    for entity in entity_list:
        mapped_entities[entity] = named_entity_map.get(entity, entity)
    return mapped_entities


# -----------------------------
# Output Structures
# -----------------------------

@dataclass
class QueryProcessingResult:
    original_query: str
    detected_language: str  # 'bn'|'en'|'mixed'|'unknown'
    normalized_query: str
    translated_query: Optional[str]
    expanded_queries: Dict[str, List[str]]  # keys: 'bn','en'
    named_entities: List[str]
    mapped_entities: Dict[str, str]
    timings_ms: Dict[str, float]
    debug: Dict[str, Any]

    retrieval_queries: Dict[str, List[str]]
    retrieval_keywords: Dict[str, List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------
# Query Processor (Module B)
# -----------------------------

class QueryProcessor:
    """
    Main entrypoint for Module B.
    """

    def __init__(
        self,
        enable_stopwords: bool = False,
        enable_wordnet_expansion: bool = False,
        translator: Optional[Translator] = None,
        named_entity_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.enable_stopwords = enable_stopwords
        self.enable_wordnet_expansion = enable_wordnet_expansion
        self.translator = translator if translator is not None else Translator()
        self.named_entity_map = named_entity_map if named_entity_map is not None else build_named_entity_mapping()

    def process(self, user_query: str) -> QueryProcessingResult:
        start_time = time.perf_counter()

        detected_language = detect_language_simple(user_query)
        language_detection_time = time.perf_counter()

        primary_language = detected_language if detected_language in ("bn", "en") else ("bn" if contains_bengali_script(user_query) else "en")
        normalized_query = normalize_query(user_query, primary_language)

        normalization_time = time.perf_counter()

        # tokens for expansion (stopword removal depends on self.enable_stopwords)
        primary_tokens = tokenize_simple(normalized_query)
        primary_tokens = remove_stopwords(primary_tokens, primary_language, self.enable_stopwords)

        translated_query_text: Optional[str] = None
        translation_start_time = time.perf_counter()

        try:
            if primary_language == "bn":
                translated_query_text = self.translator.translate(normalized_query, "bn", "en")
            elif primary_language == "en":
                translated_query_text = self.translator.translate(normalized_query, "en", "bn")
            else:
                target_language = "en" if primary_language == "bn" else "bn"
                translated_query_text = self.translator.translate(normalized_query, primary_language, target_language)  # type: ignore[arg-type]
            translation_error = None
        except TranslationError as translation_exception:
            translated_query_text = None
            translation_error = str(translation_exception)

        translation_end_time = time.perf_counter()

        expanded_primary_tokens = expand_query_tokens(
            primary_tokens,
            primary_language,
            use_wordnet=self.enable_wordnet_expansion
        )
        expanded_primary_query = " ".join(expanded_primary_tokens).strip()

        translated_language = "en" if primary_language == "bn" else "bn"
        expanded_translated_query: Optional[str] = None
        normalized_translated_query: Optional[str] = None

        if translated_query_text:
            normalized_translated_query = normalize_query(translated_query_text, translated_language)
            translated_tokens = tokenize_simple(normalized_translated_query)
            translated_tokens = remove_stopwords(translated_tokens, translated_language, self.enable_stopwords)

            expanded_translated_tokens = expand_query_tokens(
                translated_tokens,
                translated_language,
                use_wordnet=self.enable_wordnet_expansion
            )
            expanded_translated_query = " ".join(expanded_translated_tokens).strip()

        expansion_time = time.perf_counter()

        named_entities = extract_named_entities(user_query, detected_language if detected_language != "unknown" else primary_language)
        mapped_named_entities = map_named_entities(named_entities, self.named_entity_map)

        named_entity_time = time.perf_counter()

        expanded_queries: Dict[str, List[str]] = {"bn": [], "en": []}

        if primary_language == "bn":
            expanded_queries["bn"].append(normalized_query)
            if expanded_primary_query and expanded_primary_query != normalized_query:
                expanded_queries["bn"].append(expanded_primary_query)

            if translated_query_text:
                expanded_queries["en"].append(normalize_query(translated_query_text, "en"))
            if expanded_translated_query:
                expanded_queries["en"].append(expanded_translated_query)

        elif primary_language == "en":
            expanded_queries["en"].append(normalized_query)
            if expanded_primary_query and expanded_primary_query != normalized_query:
                expanded_queries["en"].append(expanded_primary_query)

            if translated_query_text:
                expanded_queries["bn"].append(normalize_query(translated_query_text, "bn"))
            if expanded_translated_query:
                expanded_queries["bn"].append(expanded_translated_query)

        else:
            expanded_queries["bn"].append(normalize_query(user_query, "bn"))
            expanded_queries["en"].append(normalize_query(user_query, "en"))
            if translated_query_text:
                expanded_queries[translated_language].append(normalize_query(translated_query_text, translated_language))

        # Add mapped entities (for both languages)
        if mapped_named_entities:
            mapped_terms: List[str] = []
            for original_entity, mapped_entity in mapped_named_entities.items():
                if original_entity != mapped_entity:
                    mapped_terms.extend([original_entity, mapped_entity])
                else:
                    mapped_terms.append(original_entity)

            normalized_mapped_terms = [
                normalize_query(term, "bn" if contains_bengali_script(term) else "en")
                for term in mapped_terms
            ]
            normalized_mapped_terms = [term for term in normalized_mapped_terms if term]

            if normalized_mapped_terms:
                bengali_mapped_terms = [term for term in normalized_mapped_terms if contains_bengali_script(term)]
                english_mapped_terms = [term for term in normalized_mapped_terms if contains_latin_script(term)]

                if bengali_mapped_terms:
                    expanded_queries["bn"].append(" ".join(bengali_mapped_terms).strip())
                if english_mapped_terms:
                    expanded_queries["en"].append(" ".join(english_mapped_terms).strip())

        # De-duplicate expanded queries
        for language_key in ("bn", "en"):
            seen_query_strings = set()
            cleaned_query_list = []
            for candidate_query in expanded_queries[language_key]:
                candidate_query = (candidate_query or "").strip()
                if not candidate_query:
                    continue
                if candidate_query in seen_query_strings:
                    continue
                cleaned_query_list.append(candidate_query)
                seen_query_strings.add(candidate_query)
            expanded_queries[language_key] = cleaned_query_list

        # Build retrieval queries and keywords for Module C
        retrieval_queries: Dict[str, List[str]] = {"bn": [], "en": []}

        # Use only clean normalized query per language (NO token-soup)
        if expanded_queries["en"]:
            retrieval_queries["en"].append(expanded_queries["en"][0])
        if expanded_queries["bn"]:
            retrieval_queries["bn"].append(expanded_queries["bn"][0])

        # Add entity-only query as a second retrieval variant (useful for location/person)
        if mapped_named_entities:
            mapped_bn_terms: List[str] = []
            mapped_en_terms: List[str] = []
            for _, mapped_value in mapped_named_entities.items():
                mapped_text = str(mapped_value)
                if contains_bengali_script(mapped_text):
                    mapped_bn_terms.append(normalize_query(mapped_text, "bn"))
                elif contains_latin_script(mapped_text):
                    mapped_en_terms.append(normalize_query(mapped_text, "en"))

            if mapped_bn_terms:
                retrieval_queries["bn"].append(" ".join(mapped_bn_terms).strip())
            if mapped_en_terms:
                retrieval_queries["en"].append(" ".join(mapped_en_terms).strip())

        # Dedup retrieval queries and cap
        for language_key in ("bn", "en"):
            seen = set()
            unique_list: List[str] = []
            for q in retrieval_queries[language_key]:
                q = (q or "").strip()
                if not q or q in seen:
                    continue
                unique_list.append(q)
                seen.add(q)
            retrieval_queries[language_key] = unique_list[:5]

        # Extract keywords from retrieval queries (hard stopword removal)
        retrieval_keywords = {
            "en": extract_keywords_for_retrieval(" ".join(retrieval_queries["en"]), "en", min_len=3),
            "bn": extract_keywords_for_retrieval(" ".join(retrieval_queries["bn"]), "bn", min_len=2),
        }

        end_time = time.perf_counter()

        timings_ms = {
            "language_detection": (language_detection_time - start_time) * 1000.0,
            "normalization": (normalization_time - language_detection_time) * 1000.0,
            "translation": (translation_end_time - translation_start_time) * 1000.0,
            "expansion": (expansion_time - translation_end_time) * 1000.0,
            "ner_and_mapping": (named_entity_time - expansion_time) * 1000.0,
            "total": (end_time - start_time) * 1000.0,
        }

        debug: Dict[str, Any] = {
            "primary_language": primary_language,
            "translated_language": translated_language,
            "translation_error": translation_error,
            "normalized_translated_query": normalized_translated_query,
            "primary_tokens": primary_tokens,
            "expanded_primary_tokens": expanded_primary_tokens,
        }

        return QueryProcessingResult(
            original_query=user_query,
            detected_language=detected_language,
            normalized_query=normalized_query,
            translated_query=translated_query_text,
            expanded_queries=expanded_queries,
            named_entities=named_entities,
            mapped_entities=mapped_named_entities,
            timings_ms=timings_ms,
            debug=debug,
            retrieval_queries=retrieval_queries,
            retrieval_keywords=retrieval_keywords,
        )


# -----------------------------
# Helper: retrieval queries for Module C
# -----------------------------

def build_queries_for_retrieval(processed_query_result: QueryProcessingResult, target_language: str, max_queries: int = 5) -> List[str]:
    """
    Returns a list of query strings you can feed to Module C retrieval models.
    target_language: 'bn' or 'en'
    """
    if target_language not in ("bn", "en"):
        raise ValueError("target_language must be 'bn' or 'en'")
    candidate_queries = processed_query_result.retrieval_queries.get(target_language, [])
    return candidate_queries[:max_queries]


# -----------------------------
# -----------------------------

if __name__ == "__main__":
    query_processor = QueryProcessor(enable_stopwords=True, enable_wordnet_expansion=False)

    while True:
        user_query = input("Query (empty to quit): ").strip()
        if not user_query:
            break

        processing_result = query_processor.process(user_query)
        print(json.dumps(processing_result.to_dict(), ensure_ascii=False, indent=2))
