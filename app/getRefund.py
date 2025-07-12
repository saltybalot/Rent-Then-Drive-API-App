# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import base64
import os
from fastapi import APIRouter
import hashlib
router = APIRouter()

# Replace with your secret key
PAYMONGO_SECRET_KEY = os.environ.get("PAYMONGO_SECRET_KEY")
BASE64_AUTH = base64.b64encode(f"{PAYMONGO_SECRET_KEY}:".encode()).decode()

class RefundRequest(BaseModel):
    booking_ref: str
    amount: int
    reason: str = "requested_by_customer"

@router.post("/")
def refund(refund_req: RefundRequest):
    headers = {"Authorization": f"Basic {BASE64_AUTH}"}

    # 1. Find payment matching booking ref in description (with pagination)
    payment = None
    url = "https://api.paymongo.com/v1/payments"
    params = {}
    while True:
        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch payments")
        data = resp.json()
        payments = data["data"]
        # Debug: print all reference_number and description values
        for p in payments:
            print(f"ID: {p.get('id')}, reference_number: {p['attributes'].get('reference_number')}, description: {p['attributes'].get('description')}")
        payment = next(
            (
                p for p in payments
                if (
                    p["attributes"].get("reference_number") == refund_req.booking_ref
                    or (
                        p["attributes"].get("description")
                        and refund_req.booking_ref in p["attributes"]["description"]
                    )
                )
            ),
            None
        )
        if payment:
            break
        # Pagination: check if more pages
        if data.get("has_more") and payments:
            last_id = payments[-1]["id"]
            params = {"after": last_id}
        else:
            break
    if not payment:
        raise HTTPException(status_code=404, detail="Booking not found in PayMongo")

    payment_id = payment["id"]

    # 2. Issue refund with idempotency key
    refund_payload = {
        "data": {
            "attributes": {
                "amount": refund_req.amount,
                "payment_id": payment_id,
                "reason": refund_req.reason
            }
        }
    }
    # Generate SHA256 idempotency key
    idempotency_string = f"{refund_req.booking_ref}:{refund_req.amount}"
    idempotency_key = hashlib.sha256(idempotency_string.encode()).hexdigest()
    refund_headers = {
        **headers,
        "Content-Type": "application/json",
        "Idempotency-Key": idempotency_key
    }
    refund_resp = requests.post(
        "https://api.paymongo.com/v1/refunds",
        json=refund_payload,
        headers=refund_headers
    )

    if refund_resp.status_code not in [200, 201]:
        raise HTTPException(status_code=refund_resp.status_code, detail=refund_resp.json())

    return {"status": "success", "refund": refund_resp.json()}
