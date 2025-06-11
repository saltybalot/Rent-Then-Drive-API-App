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
from fastapi import APIRouter
# import requests

router = APIRouter()

@router.get("/")
def fraud_detection():
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

    def ocr_image(image_path):
        # Path to tesseract executable (Only needed on Windows)
        pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'  # Update this path if necessary

        # Load the image
        image = cv2.imread(image_path)

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




    # Load the dataset
    df = pd.read_csv('rental_data.csv')

    # Separate features and label
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    print(df['is_fraud'].value_counts())

    # Normalize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC()])

    # Train the model
    class_weights = class_weight.compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    history = model.fit(
        X_train, y_train,
        class_weight={0: class_weights[0], 1: class_weights[1]},
        validation_split=0.2,
        epochs=30,
        batch_size=16
    )

    print(y_train.value_counts())


    # Evaluate the model
    # Evaluate on test data
    loss, accuracy, auc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.2f}, AUC: {auc:.2f}")

    # Predict on new rentals
    extracted_data = ocr_image('sample.jpg')
    # payment_data = get_paymongo_payment_data()
    payment_data = extracted_data  # Assuming we use the OCR data for prediction

    if (not payment_data or 
        'total_amount' not in payment_data or 
        'reference_number' not in payment_data or 
        'dates' not in payment_data
    ):
        raise ValueError("Invalid payment data extracted from OCR.")
    else:
        if (extracted_data['total_amount'] == payment_data['total_amount'] or 
            extracted_data['reference_number'] == payment_data['reference_number'] or 
            extracted_data['dates'] == payment_data['dates']):
            is_receipt_edited = 0
        else:
            is_receipt_edited = 1

    new_rental_df = pd.DataFrame([[extracted_data['total_amount'], 1, 2, is_receipt_edited, 0]],
        columns=['amount', 'booking_hour', 'lead_time_hours', 'is_receipt_edited', 'customer_history']
    )
    new_rental_scaled = scaler.transform(new_rental_df)
    prediction = model.predict(new_rental_scaled)

    print("Fraud Probability:", prediction[0][0])
    if prediction[0][0] > 0.5:
        return {"message": "Fraudulent transaction detected", "probability": prediction[0][0]}
    else:
        return {"message": "Transaction is likely legitimate", "probability": prediction[0][0]}


