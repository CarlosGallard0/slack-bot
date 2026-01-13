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
            print(f"DEBUG: Received event type: {body.get('event', {}).get('type', body.get('type'))}")
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
            initial_message = say("Thinking...")
            
            print(f"Processing query: {query}")
            response = self.agent.ask(query, thread_id=channel)
            
            self.app.client.chat_update(
                channel=channel,
                ts=initial_message["ts"],
                text=response
            )
        except Exception as e:
            print(f"Error in response flow: {e}")
            if 'initial_message' in locals():
                self.app.client.chat_update(
                    channel=channel,
                    ts=initial_message["ts"],
                    text=f"Sorry, I ran into an error: {e}"
                )
            else:
                say(f"Sorry, I ran into an error: {e}")

    def start(self):
        handler = SocketModeHandler(self.app, os.environ.get("SLACK_APP_TOKEN"))
        print("Slack bot is running...")
        handler.start()

    def send_message(self, channel: str, text: str):
        self.app.client.chat_postMessage(channel=channel, text=text)

