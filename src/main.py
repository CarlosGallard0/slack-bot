import os
from dotenv import load_dotenv
from src.core.agent import AgentCore
from src.providers.slack_provider import SlackProvider
from src.providers.teams_provider import TeamsProvider

def main():
    load_dotenv()

    agent = AgentCore()
    
    provider_type = os.environ.get("BOT_PROVIDER", "slack").lower()
    
    if provider_type == "slack":
        if not os.environ.get("SLACK_BOT_TOKEN") or not os.environ.get("SLACK_APP_TOKEN"):
            print("CRITICAL ERROR: SLACK_BOT_TOKEN or SLACK_APP_TOKEN is missing in .env")
            return
        bot = SlackProvider(agent)
    elif provider_type == "teams":
        bot = TeamsProvider(agent)
    else:
        print(f"CRITICAL ERROR: Unknown BOT_PROVIDER '{provider_type}'")
        return

    try:
        print(f"Starting {provider_type} bot...")
        bot.start()
    except Exception as e:
        print(f"Error starting the bot: {e}")

if __name__ == "__main__":
    main()

