from fileinput import filename
import re
from typing import List


def normalize_timeline(text: str) -> str:
    lines = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        line = re.sub(r"\*{2,}", "", line)

        line = line.lstrip("*-• ").strip()

        line = f"• {line}"

        lines.append(line)

    return "\n".join(lines)


def split_intro_and_points(text: str):
    intro_lines = []
    bullet_lines = []
    for line in text.split("\n"):
        clean = line.strip()
        if not clean:
            continue
        if clean.startswith("•"):
            bullet_lines.append(clean.lstrip("• ").strip())
        else:
            intro_lines.append(clean)
    return intro_lines, bullet_lines


def render_timeline_blocks(result: dict) -> list:
    timeline_text = normalize_timeline(result.get("timeline", ""))
    intro, bullets = split_intro_and_points(timeline_text)

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "Medical Research Timeline",
                "emoji": True,
            },
        },
        {"type": "divider"},
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
                {"type": "section", "text": {"type": "mrkdwn", "text": f"• {line}"}}
            )

    limitations = result.get("limitations", "")
    if limitations:
        blocks.append({"type": "divider"})
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"⚠️ *Limitations:* {limitations}"}
                ],
            }
        )

    return blocks


def format_sources_blocks(sources: List[str]) -> list:
    if not sources:
        return []

    fields = []
    for i, source in enumerate(sources[:10], 1):
        filename = source.split("/")[-1] if "/" in source else source
        display_name = filename
        fields.append({"type": "mrkdwn", "text": f"*{i}.* `{display_name}`"})

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

    blocks = format_sources_blocks(sources)
    say(
        channel=channel,
        thread_ts=thread_ts,
        blocks=blocks,
        text=f"📚 Sources ({len(sources)})",
    )
