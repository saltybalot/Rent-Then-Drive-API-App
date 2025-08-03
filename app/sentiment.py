import nltk
nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer
from fastapi import APIRouter
import json
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

router = APIRouter()
sia = SentimentIntensityAnalyzer()

# Rename to avoid conflict with FastAPI's Request
class SentimentRequest(BaseModel):
    text: str
    rating: int  # From 1 to 5

def get_expected_sentiment(rating: int) -> str:
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"

@router.post("/")
async def analyze_sentiment(request: SentimentRequest):
    input_text = request.text
    input_rating = request.rating

    input_score = sia.polarity_scores(input_text)
    input_compound = input_score['compound']
    predicted_sentiment = (
        "Positive" if input_compound >= 0.05 else
        "Negative" if input_compound <= -0.05 else
        "Neutral"
    )

    expected_sentiment = get_expected_sentiment(input_rating)
    match = predicted_sentiment == expected_sentiment

    return {
        "input": {
            "text": input_text,
            "rating": input_rating,
            "expected_sentiment": expected_sentiment,
            "predicted_sentiment": predicted_sentiment,
            "compound": input_compound,
            "match": match
        }
    }

