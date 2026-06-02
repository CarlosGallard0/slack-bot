import re


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(
        r"\b(?:[a-z]\s){2,}[a-z]\b",
        lambda m: m.group(0).replace(" ", ""),
        text,
    )

    return text
