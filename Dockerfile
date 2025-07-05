# Base image
FROM python:3.12

# Prevent prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install required OS packages
RUN apt-get update && apt-get install -y \
    libgl1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy app files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set default port
ENV PORT=8000

# Expose port (adjust if needed)
EXPOSE $PORT

# Command to run the app (adjust your module path if needed)
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
