import re
from typing import Optional, Tuple
from raptor_ingestion.normalization import normalize_text

SECTION_PATTERNS = {
    "abstract": r"\babstract\b",
    "background": r"\bbackground\b",
    "introduction": r"\bintroduction\b",
    "purpose": r"\bpurpose\b",
    "aims": r"\baims\b",
}

COMPILED = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in SECTION_PATTERNS.items()
}


def extract_section_context(
    full_text: str,
    first_page_text: Optional[str],
    max_chars: int = 2500,
) -> Tuple[Optional[str], Optional[str]]:

    if not full_text:
        return None, None

    normalized = normalize_text(full_text)

    for section, pattern in COMPILED.items():
        match = pattern.search(normalized)
        if match:
            start = match.end()
            context = normalized[start : start + max_chars].strip()
            if context:
                return section, context

    if first_page_text:
        return "first_page", normalize_text(first_page_text)[:max_chars]

    return None, None
