from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import base64
import os
import uuid
from datetime import datetime
from fastapi import APIRouter

router = APIRouter()

PAYMONGO_SECRET_KEY = os.environ.get("PAYMONGO_SECRET_KEY")
if not PAYMONGO_SECRET_KEY:
    print("Warning: PAYMONGO_SECRET_KEY environment variable not set. PayMongo functionality will be disabled.")
    PAYMONGO_SECRET_KEY = None


class CheckoutRequest(BaseModel):
    amount: float
    email: str
    description: str

@router.post("/")
def create_checkout_session(data: CheckoutRequest):
    if not PAYMONGO_SECRET_KEY:
        raise HTTPException(
            status_code=503, 
            detail="PayMongo service is not configured. Please set PAYMONGO_SECRET_KEY environment variable."
        )
    
    # Generate unique reference number
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    reference_number = f"RTD{timestamp}{unique_id}"

    # Compose description to always include the reference number
    description_with_ref = f"{data.description} | BookingRef: {reference_number}"
    
    # Multiply amount by 100 to convert to cents and ensure it's an integer
    amount_in_cents = int(round(data.amount * 100))
    
    url = "https://api.paymongo.com/v1/checkout_sessions"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": "Basic " + base64.b64encode(f"{PAYMONGO_SECRET_KEY}:".encode()).decode()
    }
    payload = {
        "data": {
            "attributes": {
                "billing": {
                    "email": data.email
                },
                "line_items": [
                    {
                        "amount": amount_in_cents,
                        "currency": "PHP",
                        "name": data.description,
                        "quantity": 1
                    }
                ],
                "payment_method_types": ["card"],
                "description": description_with_ref,
                "send_email_receipt": True,
                "show_description": True,
                "show_line_items": True,
                "reference_number": reference_number
            }
        }
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code not in (200, 201):
        raise HTTPException(status_code=400, detail=response.json())
    
    checkout_data = response.json()
    checkout_url = checkout_data["data"]["attributes"]["checkout_url"]
    print(response.status_code)
    print(response.text)
    return {"checkout_url": checkout_url, "reference_number": reference_number}
