import math
import re
from collections import Counter

DEFAULT_THRESHOLD = 0.5
_STEM_LEN = 5


def _tokenize(text: str) -> Counter:
    """Lowercased words cut to a 5-letter stem — a crude way to treat Russian
    inflections ("квадрат"/"квадраты"/"квадрата") as one word. Words of 1-2
    letters (prepositions, particles) carry no signal and are dropped."""
    words = re.findall(r"\w+", text.lower().replace("ё", "е"))
    return Counter(word[:_STEM_LEN] for word in words if len(word) > 2)


def _cosine_similarity(a: Counter, b: Counter) -> float:
    common = set(a) & set(b)
    dot = sum(a[word] * b[word] for word in common)
    norm_a = math.sqrt(sum(count * count for count in a.values()))
    norm_b = math.sqrt(sum(count * count for count in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def max_similarity(text: str, others: list[str]) -> float:
    """Highest word-overlap cosine similarity between `text` and any of `others`.

    A free, dependency-free stand-in for embeddings: fast enough to run
    before bothering an LLM call, good enough to catch near-duplicate
    phrasing. Deeper semantic duplicates (same idea, different words) are
    left for the LLM novelty check that follows this filter.
    """
    if not others:
        return 0.0

    vec = _tokenize(text)
    return max(_cosine_similarity(vec, _tokenize(other)) for other in others)


def is_too_similar(text: str, others: list[str], threshold: float = DEFAULT_THRESHOLD) -> bool:
    return max_similarity(text, others) >= threshold
