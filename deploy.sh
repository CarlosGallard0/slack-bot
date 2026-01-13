#!/bin/bash

# GCP Cloud Run Deployment Script
set -e

# Load environment variables from .env
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
else
    echo "❌ .env file not found"
    exit 1
fi

REGION="${2:-us-west1}"
PROJECT="neurodocsdomain"
SERVICE_NAME="labs-slack-bot"

echo "🚀 Deploying to Google Cloud Run"
echo "Service: $SERVICE_NAME | Region: $REGION | Project: $PROJECT"
echo ""

# Validate required environment variables
REQUIRED_VARS=(
    "MODEL_PROVIDER"
    "MODEL_NAME"
    "MODEL_TEMPERATURE"
    "EMBEDDING_PROVIDER"
    "EMBEDDING_MODEL"
    "BOT_PROVIDER"
    "SLACK_BOT_TOKEN"
    "SLACK_APP_TOKEN"
    "SLACK_SIGNING_SECRET"
    "PROJECT_ID"
    "LOCATION"
    "VERTEX_AI_API_KEY"
    "NEO4J_URI"
    "NEO4J_USER"
    "NEO4J_PASSWORD"
    "LLM_BASE_URL"
    "LLM_API_KEY"
    "EMBEDDING_BASE_URL"
    "EMBEDDING_API_KEY"
)

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Missing: $var"
        exit 1
    fi
done

echo "✓ All environment variables validated"
echo ""

# Build environment variables string
ENV_VARS="MODEL_PROVIDER=$MODEL_PROVIDER"
ENV_VARS="$ENV_VARS,MODEL_NAME=$MODEL_NAME"
ENV_VARS="$ENV_VARS,MODEL_TEMPERATURE=$MODEL_TEMPERATURE"
ENV_VARS="$ENV_VARS,EMBEDDING_PROVIDER=$EMBEDDING_PROVIDER"
ENV_VARS="$ENV_VARS,EMBEDDING_MODEL=$EMBEDDING_MODEL"
ENV_VARS="$ENV_VARS,BOT_PROVIDER=$BOT_PROVIDER"
ENV_VARS="$ENV_VARS,SLACK_MODE=http"
ENV_VARS="$ENV_VARS,GOOGLE_GENAI_USE_VERTEXAI=$GOOGLE_GENAI_USE_VERTEXAI"
ENV_VARS="$ENV_VARS,SLACK_BOT_TOKEN=$SLACK_BOT_TOKEN"
ENV_VARS="$ENV_VARS,SLACK_APP_TOKEN=$SLACK_APP_TOKEN"
ENV_VARS="$ENV_VARS,SLACK_SIGNING_SECRET=$SLACK_SIGNING_SECRET"
ENV_VARS="$ENV_VARS,PROJECT_ID=$PROJECT_ID"
ENV_VARS="$ENV_VARS,LOCATION=$LOCATION"
ENV_VARS="$ENV_VARS,VERTEX_AI_API_KEY=$VERTEX_AI_API_KEY"
ENV_VARS="$ENV_VARS,NEO4J_URI=$NEO4J_URI"
ENV_VARS="$ENV_VARS,NEO4J_USER=$NEO4J_USER"
ENV_VARS="$ENV_VARS,NEO4J_PASSWORD=$NEO4J_PASSWORD"
ENV_VARS="$ENV_VARS,LLM_BASE_URL=$LLM_BASE_URL"
ENV_VARS="$ENV_VARS,LLM_CHOICE=$LLM_CHOICE"
ENV_VARS="$ENV_VARS,LLM_API_KEY=$LLM_API_KEY"
ENV_VARS="$ENV_VARS,EMBEDDING_BASE_URL=$EMBEDDING_BASE_URL"
ENV_VARS="$ENV_VARS,EMBEDDING_API_KEY=$EMBEDDING_API_KEY"

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --region $REGION \
    --project=$PROJECT \
    --allow-unauthenticated \
    --timeout=3600 \
    --memory=1Gi \
    --cpu=1 \
    --concurrency=50 \
    --min-instances=0 \
    --max-instances=4 \
    --no-cpu-throttling \
    --set-env-vars="$ENV_VARS" \
    --source .

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Deployment successful!"
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
        --region $REGION \
        --project=$PROJECT \
        --format='value(status.url)')
    
    echo "URL: $SERVICE_URL"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Update Slack Event URL: $SERVICE_URL/slack/events"
    echo "2. Test the bot in Slack"
    echo "3. Monitor logs: gcloud run logs read $SERVICE_NAME --region $REGION"
else
    echo "❌ Deployment failed"
    exit 1
fi
