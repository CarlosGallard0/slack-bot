import re
from datetime import datetime

CURRENT_YEAR = datetime.now().year
MIN_YEAR = 1900
MAX_YEAR = CURRENT_YEAR

DOI_REGEX = re.compile(
    r"\b10\.\d{4,9}/[-._;()/:a-z0-9]+\b",
    re.IGNORECASE,
)

YEAR_PATTERN = r"(?:19\d{2}|20\d{2})"

DATE_PATTERNS = [
    (
        "published",
        re.compile(
            rf"""
            (?:published|online\s+publish-ahead-of-print)
            [^\d]{{0,20}}
            (?:
                ({YEAR_PATTERN})\b |
                \d{{1,2}}[/\-]\d{{1,2}}[/\-]({YEAR_PATTERN}) |
                \d{{1,2}}\s+[A-Za-z]+\s+({YEAR_PATTERN})
            )
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    (
        "accepted",
        re.compile(
            rf"""
            accepted
            [^\d]{{0,20}}
            (?:
                ({YEAR_PATTERN})\b |
                \d{{1,2}}[/\-]\d{{1,2}}[/\-]({YEAR_PATTERN}) |
                \d{{1,2}}\s+[A-Za-z]+\s+({YEAR_PATTERN})
            )
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
]


def extract_doi(text: str) -> str | None:
    if not isinstance(text, str):
        return None

    match = DOI_REGEX.search(text)
    return match.group(0).lower().strip() if match else None


def extract_publication_year(text: str) -> int | None:

    if not isinstance(text, str):
        return None

    for _, regex in DATE_PATTERNS:
        match = regex.search(text)
        if not match:
            continue

        for group in match.groups():
            if group is None:
                continue

            year = int(group)
            if MIN_YEAR <= year <= MAX_YEAR:
                return year

    return None
