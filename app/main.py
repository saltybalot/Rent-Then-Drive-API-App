from fastapi import FastAPI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from . import scheduler, sentiment, fraud, paymongocheckout


app = FastAPI()

app.include_router(scheduler.router, prefix="/api/scheduler")
app.include_router(sentiment.router, prefix="/api/sentiment")
app.include_router(fraud.router, prefix="/api/fraud")
app.include_router(paymongocheckout.router, prefix="/api/paymongocheckout")

