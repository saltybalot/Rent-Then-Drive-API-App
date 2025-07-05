# Environment Setup Guide

This guide will help you set up the required environment variables for the Rent-Then-Drive API App.

## Required Environment Variables

### 1. PayMongo Configuration

**PAYMONGO_SECRET_KEY**: Your PayMongo secret key for payment processing.

**To get your PayMongo secret key:**

1. Go to [PayMongo Dashboard](https://dashboard.paymongo.com/settings/keys)
2. Sign up or log in to your account
3. Navigate to Settings > API Keys
4. Copy your Secret Key (starts with `sk_`)

### 2. Firebase Configuration

**FIREBASE_CREDENTIALS_PATH**: Path to your Firebase service account JSON file.

**To get your Firebase credentials:**

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select your project (or create a new one)
3. Go to Project Settings > Service Accounts
4. Click "Generate new private key"
5. Download the JSON file
6. Place it in your project directory (e.g., `firebase-credentials.json`)

## Setup Instructions

### Option 1: Using .env file (Recommended)

1. Create a `.env` file in the project root directory
2. Add the following content:

```env
# PayMongo Configuration
PAYMONGO_SECRET_KEY=sk_your_paymongo_secret_key_here

# Firebase Configuration
FIREBASE_CREDENTIALS_PATH=./firebase-credentials.json
```

### Option 2: Using Environment Variables

Set the environment variables directly in your system:

**Windows (PowerShell):**

```powershell
$env:PAYMONGO_SECRET_KEY="sk_your_paymongo_secret_key_here"
$env:FIREBASE_CREDENTIALS_PATH="C:\path\to\your\firebase-credentials.json"
```

**Windows (Command Prompt):**

```cmd
set PAYMONGO_SECRET_KEY=sk_your_paymongo_secret_key_here
set FIREBASE_CREDENTIALS_PATH=C:\path\to\your\firebase-credentials.json
```

**Linux/Mac:**

```bash
export PAYMONGO_SECRET_KEY="sk_your_paymongo_secret_key_here"
export FIREBASE_CREDENTIALS_PATH="/path/to/your/firebase-credentials.json"
```

## Running the App

After setting up the environment variables, you can run the app:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Notes

- The app will run without Firebase functionality if `FIREBASE_CREDENTIALS_PATH` is not set
- The PayMongo checkout endpoint will return a 503 error if `PAYMONGO_SECRET_KEY` is not set
- Make sure to keep your secret keys secure and never commit them to version control
- The `.env` file should be added to your `.gitignore` file

## Testing

Once the app is running, you can:

1. Visit `http://localhost:8000/docs` for the interactive API documentation
2. Test the endpoints using the Swagger UI
3. Check the console output for any configuration warnings
