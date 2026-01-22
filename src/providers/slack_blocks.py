import re
from typing import List, Tuple


def normalize_timeline(text: str) -> str:
    """
    Normalize raw timeline text into Slack-friendly bullet format.
    """
    bullets = []

    for line in text.splitlines():
        clean = line.strip()
        if not clean:
            continue

        clean = re.sub(r"\*{2,}", "", clean)
        clean = clean.lstrip("*-• ").strip()
        bullets.append(f"• {clean}")

    return "\n".join(bullets)


def split_intro_and_bullets(text: str) -> Tuple[List[str], List[str]]:
    """
    Split text into intro lines and bullet lines.
    """
    intro, bullets = [], []

    for line in text.splitlines():
        clean = line.strip()
        if not clean:
            continue

        if clean.startswith("•"):
            bullets.append(clean.lstrip("• ").strip())
        else:
            intro.append(clean)

    return intro, bullets


def header_block(title: str = "Medical Research") -> dict:
    return {
        "type": "header",
        "text": {"type": "plain_text", "text": title, "emoji": True},
    }


def divider_block() -> dict:
    return {"type": "divider"}


def limitations_block(limitations: str) -> dict:
    return {
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": f"⚠️ *Limitations:* {limitations}"}],
    }


def render_timeline_blocks(result: dict) -> list:
    timeline = normalize_timeline(result.get("timeline", ""))
    intro, bullets = split_intro_and_bullets(timeline)

    blocks = [
        header_block(),
        divider_block(),
    ]

    if intro:
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"_{' '.join(intro)}_\n"},
            }
        )

    for line in bullets:
        if ":" in line:
            title, desc = line.split(":", 1)
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"📍 *{title.strip()}*\n{desc.strip()}",
                    },
                }
            )
        else:
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"• {line}"},
                }
            )

    limitations = result.get("limitations")
    if limitations:
        blocks.extend([divider_block(), limitations_block(limitations)])

    return blocks


def render_fact_blocks(result: dict) -> list:
    answer = result.get("answer", "").strip()

    blocks = [
        header_block(),
        divider_block(),
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": answer or "_No factual answer could be generated._",
            },
        },
    ]

    limitations = result.get("limitations")
    if limitations:
        blocks.extend([divider_block(), limitations_block(limitations)])

    return blocks


def format_sources_blocks(sources: List[str]) -> list:
    if not sources:
        return []

    fields = []
    for i, source in enumerate(sources[:10], start=1):
        name = source.split("/")[-1]
        fields.append({"type": "mrkdwn", "text": f"*{i}.* `{name}`"})

    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "📚 *Related Sources*"},
            "fields": fields,
        }
    ]


def post_sources_reply(say, sources: List[str], channel: str, thread_ts: str):
    if not sources:
        return

    say(
        channel=channel,
        thread_ts=thread_ts,
        blocks=format_sources_blocks(sources),
        text=f"📚 Sources ({len(sources)})",
    )
