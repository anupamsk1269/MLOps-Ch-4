# 1. Switch from 'slim' to the full image (approx 300MB larger but much more stable)
FROM python:3.10

# 1. System essentials
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Install V2 specifically (BEFORE copying your code)
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir pandas boto3 "sagemaker<3.0.0"

# 3. LOCAL VERIFICATION - The build will stop here if it fails
RUN python3 -c "import sagemaker; from sagemaker.session import Session; print('SUCCESS: Using SageMaker version', sagemaker.__version__)"

# 4. Copy your project files
COPY . .
# (Optional) Verify ingest.py is in the right place
RUN ls -R /app/app/ingest.py

# Expose the Flask port
EXPOSE 8080

# Run the app with Gunicorn for production performance
# Use the folder.file:variable mapping
# --timeout 120 gives RoBERTa time to load the 500MB ONNX file
# --preload ensures the model loads before the health check starts
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "120", "--preload", "app.main:app"]
