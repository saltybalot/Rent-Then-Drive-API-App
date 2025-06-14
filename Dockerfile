FROM python:3.12-bullseye

# Install tesseract
RUN apt-get update && \
    apt-get install -y tesseract-ocr && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose the port your FastAPI app runs on
EXPOSE 8000

# Command to run your FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
