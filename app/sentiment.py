import nltk
nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def analyze_sentiment():
    sia = SentimentIntensityAnalyzer()
    # Negative sentiment example
    #text = "I-it’s not like I liked using this product or anything... but I guess it did exactly what I needed, so whatever!"

    # Positive sentiment example
    text = "Onii-chan~! This product was super amazing, just like you! It made everything so much easier, and I think you'd love it too, ehehe~!"

    # Neutral sentiment example
    # text = "This product is okay, I guess. It works as expected, but nothing special."

    # Analyze the sentiment of the text
    print(sia.polarity_scores(text))
    # Output: {'neg': 0.526, 'neu': 0.474, 'pos': 0.0, 'compound': -0.5719}
    # The compound score is a normalized score that summarizes the overall sentiment of the text.

    score = sia.polarity_scores(text)

    compound = score['compound']
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    print(f"Sentiment: {sentiment}")
    return {"sentiment": sentiment}