#!/bin/bash

# Azure deployment script for Brain Tumor Detection App

# Variables - Update these
RESOURCE_GROUP="brain-tumor-rg"
APP_NAME="brain-tumor-detection-drdo"
LOCATION="eastus"
SKU="B2"  # B2 has 3.5GB RAM, good for PyTorch

echo "=== Azure Brain Tumor Detection Deployment ==="

# Login to Azure (if not already logged in)
echo "Checking Azure login..."
az account show > /dev/null 2>&1 || az login

# Create Resource Group
echo "Creating Resource Group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create App Service Plan (Linux, Python)
echo "Creating App Service Plan..."
az appservice plan create \
    --name "${APP_NAME}-plan" \
    --resource-group $RESOURCE_GROUP \
    --sku $SKU \
    --is-linux

# Create Web App
echo "Creating Web App..."
az webapp create \
    --resource-group $RESOURCE_GROUP \
    --plan "${APP_NAME}-plan" \
    --name $APP_NAME \
    --runtime "PYTHON:3.11"

# Configure startup command
echo "Configuring startup command..."
az webapp config set \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME \
    --startup-file "gunicorn --bind=0.0.0.0 --timeout=600 --workers=1 app.app:app"

# Enable Git LFS for deployment
echo "Configuring deployment..."
az webapp config appsettings set \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME \
    --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true

# Deploy from local Git
echo "Setting up deployment source..."
az webapp deployment source config-local-git \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP

echo ""
echo "=== Deployment Setup Complete ==="
echo "App URL: https://${APP_NAME}.azurewebsites.net"
echo ""
echo "To deploy your code, run:"
echo "  git remote add azure <git-url-from-above>"
echo "  git push azure main"
