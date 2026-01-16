import os
from dotenv import load_dotenv
from src.core.deterministic_agent.d_agent import DeterministicAgent
from src.providers.slack_provider import SlackProvider
from src.providers.slack_provider_http import SlackProviderHTTP
from src.providers.teams_provider import TeamsProvider

def main():
    load_dotenv()

    agent = DeterministicAgent()
    
    provider_type = os.environ.get("BOT_PROVIDER", "slack").lower()
    slack_mode = os.environ.get("SLACK_MODE", "socket").lower()  # socket or http
    
    if provider_type == "slack":
        if not os.environ.get("SLACK_BOT_TOKEN"):
            print("CRITICAL ERROR: SLACK_BOT_TOKEN is missing in .env")
            return
        if slack_mode == "http":
            if not os.environ.get("SLACK_SIGNING_SECRET"):
                print("CRITICAL ERROR: SLACK_SIGNING_SECRET is missing in .env (Required for HTTP mode)")
                return
            print("Starting Slack bot in HTTP mode (Cloud Run compatible)...")
            bot = SlackProviderHTTP(agent)
        else:
            if not os.environ.get("SLACK_APP_TOKEN"):
                print("CRITICAL ERROR: SLACK_APP_TOKEN is missing in .env (Required for Socket Mode)")
                return

            print("Starting Slack bot in Socket Mode...")
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

