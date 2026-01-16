import os
from aiohttp import web
from botbuilder.core import (
    TurnContext,
    BotFrameworkAdapter,
    BotFrameworkAdapterSettings,
)
from botbuilder.schema import Activity, ActivityTypes

from src.providers.base import BaseProvider
from src.core.agent import AgentCore

class TeamsProvider(BaseProvider):
    def __init__(self, agent: AgentCore):
        self.agent = agent
        self.app_id = os.environ.get("TEAMS_APP_ID", "")
        self.app_password = os.environ.get("TEAMS_APP_PASSWORD", "")
        self.settings = BotFrameworkAdapterSettings(self.app_id, self.app_password)
        self.adapter = BotFrameworkAdapter(self.settings)

    async def on_message_activity(self, turn_context: TurnContext):
        query = turn_context.activity.text
        if not query:
            return

        print(f"Teams received: {query}")
        
        await turn_context.send_activity("Thinking...")
        
        try:
            response = self.agent.ask(query)
            await turn_context.send_activity(response)
        except Exception as e:
            print(f"Error in Teams response flow: {e}")
            await turn_context.send_activity(f"Sorry, I ran into an error: {e}")

    async def handle_messages(self, request: web.Request) -> web.Response:
        """Main handler for incoming activities from Teams/Bot Framework."""
        if "application/json" in request.headers["Content-Type"]:
            body = await request.json()
        else:
            return web.Response(status=415)

        activity = Activity().deserialize(body)
        auth_header = request.headers.get("Authorization", "")

        async def logic(turn_context: TurnContext):
            if activity.type == ActivityTypes.message:
                await self.on_message_activity(turn_context)
            elif activity.type == ActivityTypes.conversation_update:
                # Handle bot being added to a conversation if needed
                pass

        try:
            response = await self.adapter.process_activity(activity, auth_header, logic)
            if response:
                return web.json_response(data=response.body, status=response.status)
            return web.Response(status=201)
        except Exception as e:
            print(f"Error processing activity: {e}")
            return web.Response(status=500)

    def start(self):
        """Start the aiohttp server to listen for Teams activities."""
        app = web.Application()
        app.router.add_post("/api/messages", self.handle_messages)
        
        # In cloud environments (Azure, GCP), the port is often provided by the PORT env var.
        # Teams Toolkit typically defaults to 3978 for local dev.
        port = int(os.environ.get("PORT", os.environ.get("TEAMS_PORT", 3978)))
        
        # Use 0.0.0.0 to listen on all interfaces, essential for cloud deployments.
        host = "0.0.0.0"
        print(f"Teams bot listening on http://{host}:{port}/api/messages")
        
        web.run_app(app, host=host, port=port)

    def send_message(self, channel: str, text: str):
        """Send a proactive message. (Simplified for now)"""
        print(f"TEAMS PROACTIVE [Channel: {channel}]: {text}")
