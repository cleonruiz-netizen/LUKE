FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    tesseract-ocr \
    libtesseract-dev \
    fonts-liberation \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default environment variable
ENV RUN_MODE=api

# Expose port for API (scheduler will ignore this)
EXPOSE 10000

# Run command switches based on RUN_MODE
CMD if [ "$RUN_MODE" = "scheduler" ]; then \
        python scheduler.py; \
    else \
        uvicorn main:app --host 0.0.0.0 --port 10000; \
    fi
