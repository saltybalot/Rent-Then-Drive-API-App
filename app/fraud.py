# fraud_detection.py

# 1. Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight
import tensorflow as tf
import numpy as np
import cv2
import pytesseract
from PIL import Image
import re
from fastapi import APIRouter, File, UploadFile
import joblib
# import requests

router = APIRouter()


# def get_paymongo_payment_data():
#     # secret_key = 'sk_test_xxx'  # Replace with your PayMongo secret key
#     # url = 'https://api.paymongo.com/v1/payments'

#     headers = {
#         'Authorization': f'Basic {base64_encode_key(secret_key)}',
#         'Content-Type': 'application/json'
#     }

#     response = requests.get(url, headers=headers)
    
#     if response.status_code == 200:
#         data = response.json()
#         for payment in data['data']:
#             attributes = payment['attributes']
#             print(f"Amount: {attributes['amount'] / 100:.2f} PHP")
#             print(f"Status: {attributes['status']}")
#             print(f"Reference ID: {payment['id']}")
#             print(f"Paid At: {attributes.get('paid_at')}")
#             print('-' * 40)
#         return data
#     else:
#         print("Failed to retrieve data:", response.status_code, response.text)
#         return None

def base64_encode_key(secret_key):
    import base64
    key_bytes = f"{secret_key}:".encode('ascii')
    base64_bytes = base64.b64encode(key_bytes)
    return base64_bytes.decode('ascii')

def ocr_image(image_bytes):
    # Path to tesseract executable (Only needed on Windows)
    pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'  # Update this path if necessary

    # Convert bytes to numpy array
    npimg = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Convert to RGB (OpenCV loads images in BGR format)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use pytesseract to extract text
    extracted_text = pytesseract.image_to_string(rgb_image)

    # Print the output
    print("Extracted Text:")
    print(extracted_text) 

    # Extract Total Amount Sent
    total_amount_match = re.search(r'Total Amount Sent\s+P?[\s]?(\d[\d\s.,]*)', extracted_text, re.IGNORECASE)
    total_amount = total_amount_match.group(1).replace(" ", "") if total_amount_match else "Not found"

    # Extract Reference Number
    ref_no_match = re.search(r'Ref\s*No\.\s*(\d+)', extracted_text, re.IGNORECASE)
    ref_no = ref_no_match.group(1) if ref_no_match else "Not found"

    # Regex pattern for "Month Day, Year Hour:Minute AM/PM"
    date_match = re.search(r'([A-Za-z]+ \d{1,2}, \d{4} \d{1,2}:\d{2} [AP]M)', extracted_text)
    dates = date_match.group(1) if date_match else "Not found"

    print(dates)
    print("Total Amount Sent:", total_amount)
    print("Reference Number:", ref_no)

    # Return the extracted text for further processing if needed
    extracted_text = {
        "total_amount": total_amount,
        "reference_number": ref_no,
        "dates": dates
    }

    return extracted_text

@router.post("/fraud-detection")
async def fraud_detection(image: UploadFile = File(...)):
    content = await image.read()
    # content = open('sample.jpg', 'rb').read()  # Uncomment for testing with a local file
    extracted_data = ocr_image(content)

    # Load the pre-trained model and scaler
    model = tf.keras.models.load_model('fraud_model.h5')
    scaler = joblib.load('scaler.pkl')

    if (not extracted_data or 
        'total_amount' not in extracted_data or 
        'reference_number' not in extracted_data or 
        'dates' not in extracted_data):
        raise ValueError("Invalid payment data extracted from OCR.")

    is_receipt_edited = 0  # or your logic here

    new_rental_df = pd.DataFrame([[extracted_data['total_amount'], 1, 2, is_receipt_edited, 0]],
        columns=['amount', 'booking_hour', 'lead_time_hours', 'is_receipt_edited', 'customer_history']
    )
    new_rental_scaled = scaler.transform(new_rental_df)
    prediction = model.predict(new_rental_scaled)

    prob = float(prediction[0][0])
# print(f"Prediction Probability: {prob}") // Uncomment for debugging
    return {
        "message": "Fraudulent transaction detected" if prob > 0.5 else "Transaction is likely legitimate",
        "probability": prob
    }


