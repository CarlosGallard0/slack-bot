import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from src.providers.base import BaseProvider
from src.providers.slack_blocks import (
    render_timeline_blocks,
    post_sources_reply,
)


class SlackProvider(BaseProvider):
    def __init__(self, agent):
        self.agent = agent
        self.app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

        @self.app.event("app_mention")
        def handle_mentions(event, say):
            text = event.get("text")
            if not text:
                return

            query = str(text).split(">")[-1].strip()
            self.respond_with_thinking(event["channel"], query, say)

        @self.app.message("")
        def handle_message(message, say):
            if message.get("channel_type") == "im":
                text = message.get("text")
                if not text:
                    return

                self.respond_with_thinking(message["channel"], str(text), say)

    def respond_with_thinking(self, channel, query, say):
        try:
            initial_message = say(
                channel=channel,
                text="Thinking...",
                blocks=[
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "image",
                                "image_url": "https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHJueGZ4ZzRyeGZ4ZzRyeGZ4ZzRyeGZ4ZzRyeGZ4ZzRyeGZ4JmVwPXYxX2ludGVybmFsX2dpZl9ieV9pZCZjdD1n/3o7bu3XilJ5BOiSGic/giphy.gif",
                                "alt_text": "thinking",
                            },
                            {
                                "type": "mrkdwn",
                                "text": "_Analyzing medical databases..._",
                            },
                        ],
                    }
                ],
            )

            response = self.agent.ask(query, thread_id=channel)
            if "timeline" in response:
                blocks = render_timeline_blocks(response)

                self.app.client.chat_update(
                    channel=channel,
                    ts=initial_message["ts"],
                    blocks=blocks,
                )

                sources = response.get("sources", [])
                if sources:
                    post_sources_reply(
                        say=say,
                        sources=sources,
                        channel=channel,
                        thread_ts=initial_message["ts"],
                    )

        except Exception as e:
            print(f"Error in SlackProvider: {e}")
            if "initial_message" in locals():
                self.app.client.chat_update(
                    channel=channel,
                    ts=initial_message["ts"],
                    text=f"Sorry, I ran into an error: {e}",
                )
            else:
                say(f"Sorry, I ran into an error: {e}")

    def start(self):
        handler = SocketModeHandler(
            self.app,
            os.environ.get("SLACK_APP_TOKEN"),
        )
        print("Slack bot is running...")
        handler.start()

    def send_message(self, channel: str, text: str):
        self.app.client.chat_postMessage(channel=channel, text=text)
