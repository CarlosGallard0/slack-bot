import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from src.providers.base import BaseProvider
from src.core.agent import AgentCore


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

            # 2. Update with a rich UI response
            self.app.client.chat_update(
                channel=channel,
                ts=initial_message["ts"],
                text="Here is your answer",  # Fallback text
                blocks=[
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "Query Result"},
                    },
                    {"type": "section", "text": {"type": "mrkdwn", "text": response}},
                    {"type": "divider"},
                    {
                        "type": "context",
                        "elements": [{"type": "mrkdwn", "text": f"Query: _{query}_"}],
                    },
                ],
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
