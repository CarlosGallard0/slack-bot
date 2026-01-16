# 🚀 GCP Cloud Run Deployment - Quick Start

## In 5 Minutes

```bash
# 1. Set your secrets
export SLACK_BOT_TOKEN="xoxb-your-token-here"
export SLACK_APP_TOKEN="xapp-your-token-here"
export SLACK_SIGNING_SECRET="your-signing-secret-here"
export OPENAI_API_KEY="sk-your-key-here"

# 2. Deploy
cd /Users/carlos/Desktop/Projects/slack-bot
./deploy.sh prod us-west1

# 3. After deployment, you'll get a URL like:
# https://labs-github-bot-xxxxx-uw.a.run.app

# 4. Update Slack app configuration:
# Go to api.slack.com/apps → [Your App] → Socket Mode
# - DISABLE Socket Mode (This is required for Cloud Run)
#
# Go to api.slack.com/apps → Event Subscriptions
# - Request URL: https://labs-github-bot-xxxxx-uw.a.run.app/slack/events


# 5. Test in Slack - mention your bot!
```

## Manual Deployment (if deploy.sh fails)

```bash
# Authenticate
gcloud auth login
gcloud config set project neurodocsdomain

# Deploy
gcloud run deploy labs-github-bot \
    --region us-west1 \
    --project=neurodocsdomain \
    --allow-unauthenticated \
    --timeout=3600 \
    --set-env-vars=\
BOT_PROVIDER=slack,\
SLACK_MODE=http,\
MODEL_PROVIDER=openai,\
MODEL_NAME=gpt-4,\
SLACK_BOT_TOKEN=$SLACK_BOT_TOKEN,\
SLACK_APP_TOKEN=$SLACK_APP_TOKEN,\
OPENAI_API_KEY=$OPENAI_API_KEY \
    --source .
```

## Check Status

```bash
# View service
gcloud run services describe labs-github-bot --region us-west1

# Get URL
gcloud run services describe labs-github-bot --region us-west1 --format='value(status.url)'

# See logs
gcloud run logs read labs-github-bot --region us-west1 --follow

# Test health
curl $(gcloud run services describe labs-github-bot --region us-west1 --format='value(status.url)')/health
```

## Using Vertex AI Instead of OpenAI

```bash
export MODEL_PROVIDER="vertexai"
export MODEL_NAME="gemini-pro"
export PROJECT_ID="neurodocsdomain"
export LOCATION="us-west1"

./deploy.sh prod us-west1
```

## Files Created

```
slack-bot/
├── Dockerfile                    ← NEW: Container config
├── .dockerignore                 ← NEW: Docker exclusions
├── requirements.txt              ← UPDATED: Added missing deps
├── src/
│   ├── main.py                   ← UPDATED: HTTP mode support
│   └── providers/
│       └── slack_provider_http.py ← NEW: Cloud Run provider
├── GCP_DEPLOYMENT_GUIDE.md       ← NEW: Detailed guide
├── DEPLOYMENT_CHECKLIST.md       ← NEW: Step-by-step checklist
├── DEPLOYMENT_SUMMARY.md         ← NEW: Overview
├── DEPLOYMENT_QUICK_START.md     ← NEW: This file
└── deploy.sh                     ← NEW: Automated script
```

## What Was Fixed

| Issue                                   | Fix                                                                     |
| --------------------------------------- | ----------------------------------------------------------------------- |
| Missing dependencies                    | Added: deepagents, langgraph, langchain-text-splitters, gunicorn, flask |
| No Docker support                       | Created Dockerfile & .dockerignore                                      |
| Socket Mode incompatible with Cloud Run | Created HTTP-based Slack provider                                       |
| No deployment automation                | Created deploy.sh & comprehensive guides                                |
| No health check                         | Added /health endpoint                                                  |

## Environment Variables Needed

**Always Required:**

- `SLACK_BOT_TOKEN` - From Slack app (starts with xoxb-)
- `SLACK_APP_TOKEN` - From Slack app (starts with xapp-)

**For OpenAI (default):**

- `OPENAI_API_KEY` - From platform.openai.com

**For Vertex AI (optional):**

- (Uses service account by default, no API key needed)

## Cloud Run CPU Allocation (Critical)
Since the bot processes messages in the background, you **MUST** enable "CPU is always allocated" to prevent your bot from pausing after sending the "Thinking..." message.

Update your service:
```bash
gcloud run services update labs-github-bot \
  --project=neurodocsdomain \
  --region=us-west1 \
  --no-cpu-throttling
```

## Troubleshooting

| Problem              | Solution                                                 |
| -------------------- | -------------------------------------------------------- |
| "gcloud not found"   | Install: https://cloud.google.com/sdk/docs/install       |
| "Missing API key"    | Set OPENAI_API_KEY before running deploy.sh              |
| "Bot not responding" | Check: `gcloud run logs read labs-github-bot --limit=50` |
| "Connection timeout" | Already configured with 3600s timeout                    |
| "Permission denied"  | Run: `gcloud auth login`                                 |

## Monitoring

```bash
# Real-time logs (Ctrl+C to exit)
gcloud run logs read labs-github-bot --follow

# Last 100 lines
gcloud run logs read labs-github-bot --limit=100

# JSON format for processing
gcloud run logs read labs-github-bot --format=json | jq '.textPayload'

# Check service metrics
gcloud run metrics read labs-github-bot --region us-west1
```

## After Deployment

1. **Update Slack Event URL**

   - Go to https://api.slack.com/apps
   - Select your app → Event Subscriptions
   - Set Request URL to your Cloud Run URL + `/slack/events`
   - Test (Slack will send verification)
   - Save

2. **Test Bot in Slack**

   - Mention it in a channel: `@YourBotName hello`
   - Send direct message: `@YourBotName what is X?`
   - Check logs for responses

3. **Monitor Costs**
   - View in GCP Console: Billing
   - Set budget alert: `gcloud billing budgets create --billing-account=YOUR_ID ...`

## Rollback

```bash
# List previous versions
gcloud run revisions list --service=labs-github-bot --region us-west1

# Deploy previous version
gcloud run deploy labs-github-bot --revision-suffix=<timestamp> ...
```

## Delete Service

```bash
gcloud run services delete labs-github-bot --region us-west1 --project=neurodocsdomain
```

---

**Status:** ✅ All systems ready for deployment!

Next: Run `./deploy.sh prod us-west1` with your API keys set.
