# fraud_detection.py

# 1. Imports
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight
try:
    import tensorflow as tf
    import keras
except ImportError:
    print("TensorFlow not available, using fallback mode")
    tf = None
    keras = None
import numpy as np
import cv2
import pytesseract
from PIL import Image
import re
from fastapi import APIRouter, File, UploadFile, Form
import joblib
from datetime import datetime
import imagehash
import uuid

# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK (optional)
router = APIRouter()
db = None

# Load ML models once at startup
try:
    scaler = joblib.load('scaler.pkl')
    if tf is not None and keras is not None:
        model = keras.models.load_model('fraud_model.h5')
        print("ML models loaded successfully")
    else:
        model = None
        print("TensorFlow not available, ML model not loaded")
except Exception as e:
    print(f"Failed to load ML models: {e}")
    scaler = None
    model = None

try:
    firebase_creds_path = os.getenv('FIREBASE_CREDENTIALS_PATH')
    firebase_creds_json = os.getenv('FIREBASE_CREDENTIALS_JSON')
    
    cred = None
    
    # Try JSON string first (for Render deployment)
    if firebase_creds_json:
        try:
            import json
            cred_dict = json.loads(firebase_creds_json)
            cred = credentials.Certificate(cred_dict)
            print("Loaded Firebase credentials from JSON environment variable")
        except Exception as e:
            print(f"Failed to parse Firebase JSON: {e}")
    
    # Try file path if JSON not available
    if not cred:
        # If not set in environment, try default path in project root
        if not firebase_creds_path:
            firebase_creds_path = 'r-t-d-2025-firebase-adminsdk-fbsvc-d3b3fc7d37.json'
        
        if firebase_creds_path and os.path.exists(firebase_creds_path):
            print(f"Loading Firebase credentials from: {firebase_creds_path}")
            cred = credentials.Certificate(firebase_creds_path)
        else:
            print(f"Firebase credentials not found at: {firebase_creds_path}")
            print("Running without Firebase functionality.")
    
    # Initialize Firebase if credentials are available
    if cred:
        # Check if Firebase is already initialized
        try:
            firebase_admin.get_app()
            print("Firebase already initialized")
        except ValueError:
            # Initialize Firebase with explicit credentials
            firebase_admin.initialize_app(cred, {
                'projectId': 'r-t-d-2025'
            })
            print("Firebase initialized successfully")
        
        # Create Firestore client using the Firebase app
        db = firestore.client()
        print("Firestore client created successfully")
    else:
        print("No Firebase credentials available")
        print("Running without Firebase functionality.")
        
except Exception as e:
    print(f"Firebase initialization failed: {e}")
    print("Running without Firebase functionality.") 

def calculate_business_rule_score(booking_hour, lead_time_hours, customer_history, is_receipt_edited, amount):
    """
    Calculate fraud score based on business rules for high-risk scenarios
    that the ML model might miss.
    """
    risk_score = 0.0
    
    # 1. Late night/early morning bookings (high risk)
    if booking_hour < 6 or booking_hour > 22:
        risk_score += 0.3
        if booking_hour < 4 or booking_hour > 23:
            risk_score += 0.2  # Very late/early bookings
    
    # 2. Extremely short lead times (very high risk)
    if lead_time_hours < 0.5:  # Less than 30 minutes
        risk_score += 0.4
    elif lead_time_hours < 1:  # Less than 1 hour
        risk_score += 0.2
    elif lead_time_hours < 2:  # Less than 2 hours
        risk_score += 0.1
    
    # 3. No customer history (medium risk)
    if customer_history == 0:
        risk_score += 0.15
    elif customer_history <= 1:
        risk_score += 0.05
    
    # 4. Receipt edited (high risk)
    if is_receipt_edited == 1:
        risk_score += 0.3
    
    # 5. High amount transactions (medium risk)
    try:
        amount_val = float(amount) if amount != "Not found" else 0
        if amount_val > 8000:
            risk_score += 0.2
        elif amount_val > 5000:
            risk_score += 0.1
    except:
        pass
    
    # 6. Combination of multiple risk factors (bonus risk)
    risk_factors = 0
    if booking_hour < 6 or booking_hour > 22:
        risk_factors += 1
    if lead_time_hours < 1:
        risk_factors += 1
    if customer_history <= 1:
        risk_factors += 1
    if is_receipt_edited == 1:
        risk_factors += 1
    
    if risk_factors >= 3:
        risk_score += 0.2  # High risk combination bonus
    
    return min(risk_score, 1.0)  # Cap at 1.0

def preprocess_image_for_ocr(image):
    """
    Apply fast and effective preprocessing techniques for OCR
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize if image is too small (faster processing)
    height, width = gray.shape
    if width < 600:
        scale_factor = 600 / width
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Apply only the most effective preprocessing techniques
    preprocessed_images = []
    
    # 1. Original grayscale (baseline)
    preprocessed_images.append(("original", gray))
    
    # 2. Adaptive thresholding (most effective for receipts)
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    preprocessed_images.append(("adaptive_thresh", adaptive_thresh))
    
    # 3. Contrast enhancement (good for low contrast images)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    preprocessed_images.append(("enhanced", enhanced))
    
    return preprocessed_images

def ocr_image(image_bytes):
    # Path to tesseract executable
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract" # Render on Linux
    
    # Path to tesseract executable (Only needed on Windows)
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Render on Windows

    # Decode image bytes to numpy array for OpenCV
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Image decoding failed. Invalid image data.")

    # Convert to RGB for pytesseract
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess the image with multiple techniques
    preprocessed_images = preprocess_image_for_ocr(image)
    
    # Try only the most effective OCR configurations
    ocr_configs = [
        '--oem 3 --psm 6',  # LSTM OCR Engine + Uniform block of text (fastest and most accurate)
        '--oem 3 --psm 8',  # LSTM OCR Engine + Single word (good for sparse text)
    ]
    
    best_text = ""
    best_confidence = 0
    best_config = ""
    best_preprocessing = ""
    
    print("Trying OCR with different preprocessing...")
    
    # Try each preprocessing technique with the best OCR config
    for preprocess_name, preprocessed_img in preprocessed_images:
        try:
            # Use the most effective config for receipts
            config = '--oem 3 --psm 6'
            
            # Extract text
            text = pytesseract.image_to_string(preprocessed_img, config=config)
            
            print(f"Preprocessing: {preprocess_name}")
            print(f"Text: {text.strip()}")
            
            # Keep the first result with meaningful text
            if text.strip() and len(text.strip()) > 10:  # At least 10 characters
                best_text = text
                best_preprocessing = preprocess_name
                best_config = config
                break
                
        except Exception as e:
            print(f"Error with preprocessing {preprocess_name}: {e}")
            continue
    
    # If no good results, try the original image
    if not best_text.strip():
        print("Falling back to original image...")
        best_text = pytesseract.image_to_string(rgb_image, config='--oem 3 --psm 6')
        best_preprocessing = "original"
        best_config = "--oem 3 --psm 6"
    
    print(f"\nBest OCR Result:")
    print(f"Preprocessing: {best_preprocessing}")
    print(f"Config: {best_config}")
    print(f"Extracted Text:")
    print(best_text)
    
    extracted_text_raw = best_text
    
    # Simple text cleaning
    def clean_ocr_text(text):
        """Simple text cleaning for OCR results"""
        if not text:
            return ""
        
        # Remove excessive whitespace and normalize
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # Keep non-empty lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    # Clean the extracted text
    extracted_text_raw = clean_ocr_text(extracted_text_raw)
    print(f"Cleaned Text:")
    print(extracted_text_raw)
    
    # Optional: Save debug image (uncomment if needed)
    # cv2.imwrite(f"debug_{best_preprocessing}.png", preprocessed_images[0][1])

    # Initialize variables
    total_amount = "Not found"
    ref_no = "Not found"
    dates = "Not found"

    # Try PayMongo format first
    if "paymongo" in extracted_text_raw.lower() or "payment amount" in extracted_text_raw.lower():
        print("Detected PayMongo receipt format")
        
        # Extract Payment Amount (PayMongo format)
        # Look for patterns like "P 1,000.00" or "Payment amount P 1,000.00"
        paymongo_amount_patterns = [
            r'Payment amount\s*[^\d\n]?\s*([\d,]+\.?\d*)',  # Allow any non-digit (like '?', 'P', etc.) before amount
            r'Total\s+[^\d\n]?\s*([\d,]+\.?\d*)',  # Allow any non-digit before amount
            r'[^\d\n]?\s*([\d,]+\.?\d*)',  # General: any non-digit before amount
            r'Total:\s*[^\d\n]?\s*([\d,]+\.?\d*)',
            r'([\d,]+\.?\d*)\s*/ unit'
        ]
        
        for pattern in paymongo_amount_patterns:
            amount_match = re.search(pattern, extracted_text_raw, re.IGNORECASE)
            if amount_match:
                total_amount = amount_match.group(1).replace(",", "")
                print(f"DEBUG: PayMongo amount pattern matched - Extracted amount: {total_amount}")
                break
        
        # Extract Reference Number (PayMongo format)
        # Look for patterns like "RTD20250706004251b95be64b" or "cs_ZNR4tYU18A71ofPzA05n3xP5" or similar
        ref_patterns = [
            r'\bRTD[a-zA-Z0-9]{10,}\b',  # RTD followed by at least 10 alphanumeric chars
            r'\bcs_[a-zA-Z0-9_]{20,}\b',  # cs_ followed by at least 20 chars
            r'Reference\s+([A-Za-z0-9]{8,})',  # Reference followed by at least 8 chars
            r'Ref\s+([A-Za-z0-9]{8,})'  # Ref followed by at least 8 chars
        ]
        
        print("DEBUG: Searching for reference number in PayMongo receipt...")
        for i, pattern in enumerate(ref_patterns):
            ref_match = re.search(pattern, extracted_text_raw, re.IGNORECASE)
            if ref_match:
                if pattern.startswith(r'\bRTD') or pattern.startswith(r'\bcs_'):
                    ref_no = ref_match.group(0)
                else:
                    ref_no = ref_match.group(1)
                print(f"DEBUG: Pattern {i+1} matched - Extracted reference: {ref_no}")
                break
        else:
            print("DEBUG: No reference pattern matched")
        
        # If no reference number found for PayMongo, generate a placeholder
        print(f"DEBUG: Current ref_no value: '{ref_no}'")
        if ref_no == "Not found":
            import time
            timestamp = int(time.time())
            ref_no = f"PAYMONGO_{timestamp}"
            print(f"DEBUG: Generated placeholder reference number: {ref_no}")
        else:
            print(f"DEBUG: Using extracted reference number: {ref_no}")
        
        # Extract Date (PayMongo format)
        # Look for patterns like "July 3, 2025, 1:23 PM"
        date_patterns = [
            r'([A-Za-z]+ \d{1,2}, \d{4},? \d{1,2}:\d{2} [AP]M)',
            r'Payment date\s*([A-Za-z]+ \d{1,2}, \d{4},? \d{1,2}:\d{2} [AP]M)',
            r'([A-Za-z]+ \d{1,2}, \d{4})'
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, extracted_text_raw, re.IGNORECASE)
            if date_match:
                dates = date_match.group(1)
                break
        # If no date found, use current server time
        if dates == "Not found" or not dates.strip():
            dates = datetime.now().strftime("%B %d, %Y %I:%M %p")

    else:
        # Try GCash format (original logic)
        print("Detected GCash receipt format")
        
        # Extract Total Amount (GCash format) - Updated patterns
        gcash_amount_patterns = [
            r'Total Amount Sent\s+P?[\s]?(\d[\d\s.,]*)',  # Original pattern
            r'Total\s+P\s*(\d[\d\s.,]*)',  # "Total P 10.00" format
            r'Total\s*(\d[\d\s.,]*)',  # "Total 10.00" format
            r'P\s*(\d[\d\s.,]*)',  # "P 10.00" format
        ]
        
        total_amount = "Not found"
        for pattern in gcash_amount_patterns:
            amount_match = re.search(pattern, extracted_text_raw, re.IGNORECASE)
            if amount_match:
                total_amount = amount_match.group(1).replace(" ", "").replace(",", "")
                print(f"DEBUG: Amount pattern matched - Extracted amount: {total_amount}")
                break

        # Extract Reference Number (GCash format) - Updated patterns
        gcash_ref_patterns = [
            r'Ref\s*No\.\s*(\d+)',  # Original pattern
            r'(\d{8,})',  # Any 8+ digit number (common for reference numbers)
            r'Reference\s*(\d+)',  # "Reference 893641967" format
        ]
        
        ref_no = "Not found"
        for pattern in gcash_ref_patterns:
            ref_match = re.search(pattern, extracted_text_raw, re.IGNORECASE)
            if ref_match:
                ref_no = ref_match.group(1)
                print(f"DEBUG: Reference pattern matched - Extracted reference: {ref_no}")
                break

        # Extract Date (GCash format)
        date_match = re.search(r'([A-Za-z]+ \d{1,2}, \d{4} \d{1,2}:\d{2} [AP]M)', extracted_text_raw)
        dates = date_match.group(1) if date_match else "Not found"

    # Compute perceptual hash (pHash) using PIL and imagehash
    # Convert numpy array (OpenCV image) to PIL Image
    pil_image = Image.fromarray(rgb_image)
    perceptual_hash = str(imagehash.phash(pil_image))
    print("Perceptual hash:", perceptual_hash)

    # If no amount or reference found, try alternative patterns as fallback
    if total_amount == "Not found" or ref_no == "Not found":
        print("DEBUG: Primary extraction failed, trying fallback patterns...")
        
        # Try to find any amount pattern
        if total_amount == "Not found":
            fallback_amount_patterns = [
                r'(\d+\.?\d*)',  # Any decimal number
                r'P\s*(\d+\.?\d*)',  # P followed by number
                r'Total\s*(\d+\.?\d*)',  # Total followed by number
            ]
            for pattern in fallback_amount_patterns:
                amount_match = re.search(pattern, extracted_text_raw, re.IGNORECASE)
                if amount_match:
                    total_amount = amount_match.group(1)
                    print(f"DEBUG: Fallback amount pattern matched - Extracted amount: {total_amount}")
                    break
        
        # Try to find any reference number pattern
        if ref_no == "Not found":
            fallback_ref_patterns = [
                r'(\d{8,})',  # Any 8+ digit number
                r'(\d{6,})',  # Any 6+ digit number as last resort
            ]
            for pattern in fallback_ref_patterns:
                ref_match = re.search(pattern, extracted_text_raw, re.IGNORECASE)
                if ref_match:
                    ref_no = ref_match.group(1)
                    print(f"DEBUG: Fallback reference pattern matched - Extracted reference: {ref_no}")
                    break

    # Return all data together
    return {
        "total_amount": total_amount,
        "reference_number": ref_no,
        "dates": dates,
        "perceptual_hash": perceptual_hash,
        "raw_text": extracted_text_raw
    }

@router.post("/")
async def save_payment_transaction(
    image: UploadFile = File(...),
    booking_ref: str = Form(...),
    payment_method: str = Form(...),
    payment_time: str = Form(...),
    payment_proof_url: str = Form(...),
    payment_status: str = Form(...),
    booking_creation: str = Form(...),
    checking_time: str = Form(...),
    customer_history: str = Form(...)
):
    content = await image.read()
    extracted_data = ocr_image(content)

    total_amount = extracted_data.get("total_amount", 0.0)
    ref_no = extracted_data.get("reference_number", "Not found")
    img_hash = extracted_data.get("perceptual_hash", "Not found")
    ocr_date = extracted_data.get("dates", "Not found")

    # Check Firestore for duplicate ref_no or perceptual_hash across all payments
    is_duplicate_ref = False
    is_duplicate_hash = False

    if db is not None:
        ref_query = db.collection('payments').where('referenceID', '==', ref_no).stream()
        if any(ref_query):
            is_duplicate_ref = True

        hash_query = db.collection('payments').where('imageHash', '==', img_hash).stream()
        if any(hash_query):
            is_duplicate_hash = True

    is_receipt_edited = 1 if (is_duplicate_ref or is_duplicate_hash) else 0
    ocr_status = "Duplicate" if is_receipt_edited else "Pass"

    # Prepare ML input with robust datetime parsing
    def parse_datetime(date_string):
        """Parse datetime string with multiple format support"""
        formats = [
            "%m/%d/%Y %I:%M %p",  # 7/9/2025 1:07 PM
            "%m/%d/%Y %I:%M:%S %p",  # 7/9/2025 1:07:00 PM
            "%m/%d/%Y %H:%M",  # 7/9/2025 13:07
            "%m/%d/%Y %H:%M:%S",  # 7/9/2025 13:07:00
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_string, fmt)
            except ValueError:
                continue
        
        # If no format works, try to clean the string and retry
        try:
            # Remove any extra spaces and normalize
            cleaned = date_string.strip()
            return datetime.strptime(cleaned, "%m/%d/%Y %I:%M %p")
        except ValueError:
            raise ValueError(f"Unable to parse datetime: {date_string}")
    
    booking_dt = parse_datetime(booking_creation)
    checkin_dt = parse_datetime(checking_time)
    payment_dt = parse_datetime(payment_time)
    
    # Parse OCR date for timestamp
    def parse_ocr_date(ocr_date_string):
        """Parse date from OCR extraction for timestamp"""
        if ocr_date_string == "Not found":
            return payment_dt  # Fallback to payment_time from form
        
        # Try different OCR date formats
        ocr_formats = [
            "%B %d, %Y, %I:%M %p",  # July 3, 2025, 1:23 PM
            "%B %d, %Y %I:%M %p",   # July 3, 2025 1:23 PM
            "%B %d, %Y",            # July 3, 2025
        ]
        
        for fmt in ocr_formats:
            try:
                return datetime.strptime(ocr_date_string, fmt)
            except ValueError:
                continue
        
        # If OCR date parsing fails, use payment_time from form
        return payment_dt
    
    # Use OCR date for timestamp, fallback to payment_time if OCR fails
    timestamp_dt = parse_ocr_date(ocr_date)

    booking_hour = booking_dt.hour
    lead_time_hours = max(0, int((checkin_dt - booking_dt).total_seconds() / 3600))

    # Convert amount to proper numeric value
    try:
        if total_amount != "Not found":
            # Remove any non-numeric characters except decimal point
            amount_clean = re.sub(r'[^\d.]', '', str(total_amount))
            amount_numeric = float(amount_clean) if amount_clean else 0.0
        else:
            amount_numeric = 0.0
    except (ValueError, TypeError):
        amount_numeric = 0.0
    
    new_rental_df = pd.DataFrame({
        'amount': [amount_numeric],
        'booking_hour': [booking_hour],
        'lead_time_hours': [lead_time_hours],
        'is_receipt_edited': [is_receipt_edited],
        'customer_history': [customer_history]
    })

    # Calculate business rule score for high-risk scenarios
    business_rule_score = calculate_business_rule_score(
        booking_hour, lead_time_hours, int(customer_history), is_receipt_edited, amount_numeric
    )
    
    # Use pre-loaded models
    if scaler is not None and model is not None:
        new_rental_scaled = scaler.transform(new_rental_df)
        prediction = model.predict(new_rental_scaled, verbose=0)  # Add verbose=0 to suppress warnings
        ml_fraud_score = float(prediction[0][0])
        
        # Combine ML score with business rule score (weighted average)
        # Give more weight to business rules for extreme cases
        if business_rule_score > 0.5:  # High business rule risk
            fraud_score = 0.3 * ml_fraud_score + 0.7 * business_rule_score
        else:  # Normal case
            fraud_score = 0.7 * ml_fraud_score + 0.3 * business_rule_score
            
        is_fraud = fraud_score > 0.5
    else:
        # Fallback to business rules only
        fraud_score = business_rule_score
        is_fraud = fraud_score > 0.5

    # Store in Firestore payments collection (if Firebase is available)
    if db is not None:
        payment_doc_id = str(uuid.uuid4())
        
        # Extract booking document ID from the full path
        # booking_ref format: "/Bookings/romzhIDIVxSdIHKrRcxt" -> extract "romzhIDIVxSdIHKrRcxt"
        booking_doc_id = booking_ref.split('/')[-1] if '/' in booking_ref else booking_ref
        
        db.collection("payments").document(payment_doc_id).set({
            "referenceID": ref_no,
            "timestamp": timestamp_dt,  # Use OCR-extracted date
            "ocr_date_raw": ocr_date,  # Store original OCR date text for reference
            "paymentMethod": payment_method,
            "amount": amount_numeric,  # Use numeric amount
            "bookingRef": db.document(f"bookings/{booking_doc_id}"),  # doc reference
            "imageHash": img_hash,
            "fraudScore": fraud_score,
            "isFraud": is_fraud,
            "payment_status": payment_status,  # initial status, can be changed by admin
            "ocr_status": ocr_status,
            "paymentProofURL": payment_proof_url,
            "timestamp_local": timestamp_dt.strftime("%Y-%m-%d %I:%M:%S %p")  # Store as string with AM/PM
        })

    return {
        "booking_ref": booking_ref,
        "referenceID": ref_no,
        "ocr_status": ocr_status,
        "fraudScore": round(fraud_score, 3),
        "mlScore": round(ml_fraud_score, 3) if scaler is not None and model is not None else None,
        "businessRuleScore": round(business_rule_score, 3),
        "isFraud": is_fraud,
        "riskFactors": {
            "lateNightBooking": booking_hour < 6 or booking_hour > 22,
            "shortLeadTime": lead_time_hours < 1,
            "noHistory": int(customer_history) == 0,
            "receiptEdited": is_receipt_edited == 1
        },
        "payment_doc_id": payment_doc_id if db is not None else None,
        "message": "Fraudulent transaction detected" if is_fraud else "Transaction likely legitimate"
    }


