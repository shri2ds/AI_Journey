# Base Layer: Start with a lightweight Linux + Python 3.9
FROM python:3.9

# Setup: Create a folder inside the container called /app
WORKDIR /app

# Dependencies: Copy the requirement list and install libraries
# We do this BEFORE copying the code to use Docker's "Cache" (faster builds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python script and the Model into the container
# Copying everything from current folder (.) to container folder (.)
COPY . .

# Start: The command to run when the container turns on
# --host 0.0.0.0 is REQUIRED to let outside world (your browser) access it
CMD ["uvicorn", "ML_API_Deploy:app", "--host", "0.0.0.0", "--port", "80" ]
