import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from src.providers.base import BaseProvider
from src.core.agent import AgentCore
import ast
from slackstyler import SlackStyler

styler = SlackStyler()


def separate_thinking_from_response(raw_text):
    """
    Separates a Python dictionary string at the start from the rest of the text.
    Handles nested braces and quoted strings correctly.
    """
    if not raw_text.strip().startswith("{"):
        return None, raw_text

    balance = 0
    in_quote = False
    quote_char = None

    for i, char in enumerate(raw_text):
        # Toggle quote state (handle escaped quotes if necessary, simplified here)
        if char in ['"', "'"]:
            if not in_quote:
                in_quote = True
                quote_char = char
            elif char == quote_char:
                # check for escape char (simple check)
                if i > 0 and raw_text[i - 1] != "\\":
                    in_quote = False

        # Count braces only if NOT inside a string
        if not in_quote:
            if char == "{":
                balance += 1
            elif char == "}":
                balance -= 1

            # If balance hits zero, we found the end of the dict
            if balance == 0:
                dict_part = raw_text[: i + 1]
                response_part = raw_text[i + 1 :]

                try:
                    # Convert string dict to actual Python dict
                    parsed_dict = ast.literal_eval(dict_part)
                    return parsed_dict, response_part
                except (ValueError, SyntaxError):
                    # Fallback if parsing fails
                    return None, raw_text

    return None, raw_text


class SlackProvider(BaseProvider):
    def __init__(self, agent: AgentCore):
        self.agent = agent
        self.app = App(token=os.environ.get("SLACK_BOT_TOKEN"))
        self.setup_handlers()

    def setup_handlers(self):
        @self.app.middleware
        def log_request(logger, body, next):
            print(
                f"DEBUG: Received event type: {body.get('event', {}).get('type', body.get('type'))}"
            )
            return next()

        @self.app.event("app_mention")
        def handle_mentions(event, say):
            text = event.get("text")
            if not text:
                return
            if isinstance(text, list):
                text = " ".join(str(t) for t in text)
            query = str(text).split(">")[-1].strip()
            self.respond_with_thinking(event["channel"], query, say)

        @self.app.message("")
        def handle_message(message, say):
            if message.get("channel_type") == "im":
                text = message.get("text")
                if not text:
                    return
                if isinstance(text, list):
                    text = " ".join(str(t) for t in text)
                text = str(text)
                self.respond_with_thinking(message["channel"], text, say)

    def respond_with_thinking(self, channel, query, say):
        try:
            # 1. Send initial customized loading message
            initial_message = say(
                text="Thinking...",
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": ":hourglass_flowing_sand: *Thinking...* I'm looking that up for you.",
                        },
                    }
                ],
            )

            response = self.agent.ask(query)
            thought_data, clean_response = separate_thinking_from_response(response)
            thought_text = thought_data.get("thinking") if thought_data else None

            blocks = [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "Response"},
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": styler.convert(clean_response)},
                },
                {"type": "divider"},
                {
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": f"Query: _{query}_"}],
                },
            ]

            # 3. (Optional) Add thinking as a collapsible or separate context
            if thought_text:
                self.app.client.chat_postMessage(
                    channel=channel,
                    thread_ts=initial_message["ts"],  # This puts it in the thread
                    text=f"🤖 *Thinking Process:*\n{thought_text}",
                )

            self.app.client.chat_update(
                channel=channel,
                ts=initial_message["ts"],
                text="Here is your answer",  # Fallback text
                blocks=blocks,
            )
        except Exception as e:
            # Error UI
            say(text=f"Error: {e}")

    def start(self):
        handler = SocketModeHandler(self.app, os.environ.get("SLACK_APP_TOKEN"))
        print("Slack bot is running...")
        handler.start()

    def send_message(self, channel: str, text: str):
        self.app.client.chat_postMessage(channel=channel, text=text)
