import random
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from fastapi import APIRouter

# Download VADER lexicon if not already present
nltk.download("vader_lexicon")

# Initialize VADER analyzer
sia = SentimentIntensityAnalyzer()

router = APIRouter()

def get_simulated_samples():
    return {
        1: [
            "Terrible experience. The car was filthy and smelly.",
            "Very disappointed. I will never use this service again.",
            "Worst trip ever. The engine died mid-trip.",
            "Awful service. Nothing went right.",
            "Completely unacceptable. The car broke down halfway.",
            "Disgusting interior. I couldn’t even sit comfortably."
        ],
        2: [
            "Not a great ride. The AC was broken.",
            "Poor service. The car was late.",
            "It had issues, but we still managed to reach.",
            "Unreliable vehicle. I was worried the whole time.",
            "Driver was polite, but the car was poorly maintained.",
            "Below expectations. Definitely needs improvement."
        ],
        3: [
            "It was okay, nothing special.",
            "Average experience. Could be better.",
            "Not bad, not great either.",
            "Neutral experience. Just got the job done.",
            "The ride was fine, but nothing stood out.",
            "Fair service overall. Room for improvement."
        ],
        4: [
            "Good trip. The driver was professional.",
            "Nice car and smooth ride.",
            "Overall a pleasant experience.",
            "Arrived on time and the car was clean.",
            "Comfortable ride with decent service.",
            "Everything went well, just not exceptional."
        ],
        5: [
            "Excellent service! Loved the car.",
            "Superb! Will definitely book again.",
            "Amazing experience. Highly recommended!",
            "Absolutely fantastic! Smooth and professional.",
            "Loved every minute. Best rental ever.",
            "Incredible value and flawless ride!"
        ]
    }

def get_expected_sentiment(rating: int) -> str:
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"

@router.get("/")
async def test_accuracy():
    simulated_samples = get_simulated_samples()

    feedback_data = []
    for rating, texts in simulated_samples.items():
        label = get_expected_sentiment(rating)
        for text in texts:
            feedback_data.append({"text": text, "label": label})

    y_true = []
    y_pred = []

    for entry in feedback_data:
        label = entry["label"]
        score = sia.polarity_scores(entry["text"])["compound"]
        prediction = (
            "Positive" if score >= 0.05 else
            "Negative" if score <= -0.05 else
            "Neutral"
        )
        y_true.append(label)
        y_pred.append(prediction)

    accuracy = round(accuracy_score(y_true, y_pred), 2)
    report = classification_report(y_true, y_pred, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred).tolist()

    return {
        "based_on_simulated_data": True,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix
    }