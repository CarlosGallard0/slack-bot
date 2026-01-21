import os
import threading
from flask import Flask, request
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from src.providers.base import BaseProvider
from dotenv import load_dotenv
from src.providers.slack_blocks import (
    render_timeline_blocks,
    format_sources_blocks,
)

load_dotenv(override=True)

app_flask = Flask(__name__)
slack_app = None
handler = None


class SlackProviderHTTP(BaseProvider):
    """HTTP-based Slack provider for Cloud Run deployment"""

    def __init__(self, agent):
        global slack_app, handler
        self.agent = agent
        self.app = App(
            token=os.environ.get("SLACK_BOT_TOKEN"),
            signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
        )
        self.handler = SlackRequestHandler(self.app)

        slack_app = self.app
        handler = self.handler

        self.setup_handlers()

    def setup_handlers(self):

        @self.app.middleware
        def log_request(logger, body, next):
            print(
                f"DEBUG: Received event type: {body.get('event', {}).get('type', body.get('type'))}"
            )
            return next()

        @self.app.event("app_mention")
        def handle_mentions(event, say, ack):
            ack()
            text = event.get("text")
            if not text:
                return
            if isinstance(text, list):
                text = " ".join(str(t) for t in text)
            query = str(text).split(">")[-1].strip()
            self.respond_with_thinking(event["channel"], query, say)

        @self.app.message("")
        def handle_message(message, say, ack):
            ack()
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

            def process_request():
                initial_message = None
                try:
                    thinking_blocks = [
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "image",
                                    "image_url": (
                                        "https://i.giphy.com/media/"
                                        "v1.Y2lkPTc5MGI3NjExNHJueGZ4ZzRyeGZ4ZzRyeGZ4"
                                        "ZzRyeGZ4ZzRyeGZ4ZzRyeGZ4JmVwPXYxX2ludGVy"
                                        "bmFsX2dpZl9ieV9pZCZjdD1n/3o7bu3XilJ5BOiSGic/giphy.gif"
                                    ),
                                    "alt_text": "thinking",
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": "_Analyzing medical databases..._",
                                },
                            ],
                        }
                    ]

                    initial_message = say(
                        channel=channel,
                        text="Thinking...",
                        blocks=thinking_blocks,
                    )

                    print(f"Processing query: {query} in thread: {channel}")
                    response = self.agent.ask(query, thread_id=channel)

                    if isinstance(response, dict):
                        timeline_blocks = render_timeline_blocks(response)
                        sources_blocks = format_sources_blocks(
                            response.get("sources", [])
                        )

                        self.app.client.chat_update(
                            channel=channel,
                            ts=initial_message["ts"],
                            blocks=timeline_blocks,
                            text="Medical Research Timeline",
                        )

                        if sources_blocks:
                            self.app.client.chat_postMessage(
                                channel=channel,
                                thread_ts=initial_message["ts"],
                                blocks=sources_blocks,
                                text="Related Sources",
                            )

                except Exception as e:
                    print(f"Error in response flow: {e}")
                    error_msg = f"Sorry, I ran into an error: {e}"
                    if initial_message:
                        self.app.client.chat_update(
                            channel=channel, ts=initial_message["ts"], text=error_msg
                        )
                    else:
                        try:
                            say(error_msg)
                        except Exception as say_error:
                            print(f"Could not send error message: {say_error}")

            threading.Thread(target=process_request).start()

        except Exception as e:
            print(f"Error starting response flow: {e}")
            say(f"Sorry, I ran into an error: {e}")

    def start(self):
        """Start Flask server for HTTP-based event handling"""
        port = int(os.environ.get("PORT", 8080))
        print(f"Slack bot is running on port {port}...")
        app_flask.run(host="0.0.0.0", port=port, debug=False)

    def send_message(self, channel: str, text: str):
        self.app.client.chat_postMessage(channel=channel, text=text)


@app_flask.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for Cloud Run"""
    return {"status": "healthy"}, 200


@app_flask.route("/slack/events", methods=["POST"])
def slack_events():
    """Handle Slack events"""
    if handler is None:
        return {"error": "Slack handler not initialized"}, 500
    return handler.handle(request)
