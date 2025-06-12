import nltk
nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer
from fastapi import APIRouter, Request
import json

router = APIRouter()

@router.post("/")
async def analyze_sentiment(request: Request):
    # Read raw body
    raw_body = await request.body()
    try:
        # Decode and parse the stringified JSON
        decoded = raw_body.decode()
        parsed = json.loads(decoded)  # handles {"text": "..."} or '"{\"text\": \"...\"}"'
        
        # If 'text' is not in parsed, it might still be a stringified object
        if isinstance(parsed, str):
            parsed = json.loads(parsed)

        text = parsed.get("text", "")
    except Exception as e:
        return {"error": "Invalid input format", "details": str(e)}

    # Sentiment analysis
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)
    compound = score['compound']

    sentiment = (
        "Positive" if compound >= 0.05 else
        "Negative" if compound <= -0.05 else
        "Neutral"
    )

    return {
        "text": text,
        "compound": compound,
    }