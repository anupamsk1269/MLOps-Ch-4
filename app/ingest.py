import os
import pandas as pd
import boto3
import sagemaker
import logging
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup
# Setup for logging 
# --- CHAPTER 6: CLOUDWATCH LOGGING CONFIGURATION ---
logger = logging.getLogger("SageMakerIngest")
logger.setLevel(logging.INFO)

# Standardize format for CloudWatch: [Timestamp] LEVEL - Message
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')

# Stream to stdout so SageMaker picks it up
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)
# 1. Setup SageMaker Session
region = os.environ.get("AWS_REGION", "us-east-1")
boto_session = boto3.Session(region_name=region)
sagemaker_session = sagemaker.Session(boto_session=boto_session)

def run_ingestion():
    # 2. Initialize variables at the TOP to prevent NameError
    processed_records = []
    final_df = pd.DataFrame()
    
    input_data_path = "/opt/ml/processing/input/raw_data.csv"
    feature_group_name = "roberta-sentiment-features"
    logger.info(f"--- Starting Ingestion Job for Group: {feature_group_name} ---")
    # 3. Read Data
    if not os.path.exists(input_data_path):
        print(f"Error: {input_data_path} not found")
        logger.error(f"CRITICAL: Input file missing at {input_path}")
        return
    else:
        df = pd.read_csv(input_data_path)
        logger.info(f"Loaded {len(df)} rows from CSV. Starting RoBERTa processing...")
        # 4. Process Logic (Simulated or actual RoBERTa logic)
        for index, row in df.iterrows():
        # Example processing - ensure these match your model output
            if index % 100 == 0 and index > 0:
                logger.info(f"Progress: Processed {index} records...")
            record = {
            "guid": str(row.get("guid", index)),
            "event_time": pd.Timestamp.now().timestamp(),
            "sentiment": "neutral", # Replace with model prediction
            "score": 0.0            # Replace with model score
            }
            processed_records.append(record)

    # 5. Create final_df safely
    if processed_records:
        final_df = pd.DataFrame(processed_records)
        
        # 6. CRITICAL: Filter to ONLY valid features defined in Feature Group
        # This prevents the ValidationError we saw earlier
        valid_features = ["guid", "event_time", "sentiment", "score"]
        logger.info(f"Filtering DataFrame for Feature Store schema: {valid_features}")
        df_to_ingest = final_df[valid_features]

        # 7. Ingest into Feature Store
        print(f"Ingesting {len(df_to_ingest)} records into {feature_group_name}...")
        logger.info(f"Ingesting {len(df_to_ingest)} records into {feature_group_name}...")
        fg = FeatureGroup(name=feature_group_name, sagemaker_session=sagemaker_session)
        fg.ingest(data_frame=df_to_ingest, max_workers=3, wait=True)
        logger.info("SUCCESS: Data successfully landed in Feature Store.")
        print("Ingestion complete!")
    else:
        print("No records were processed.")
except Exception as e:
        # 4. The "Saving Grace" Log
        # logger.exception captures the full stack trace automatically
        logger.exception("FATAL ERROR: Ingestion failed unexpectedly.")
        raise # Re-raise to ensure the SageMaker job status marks as 'Failed'
if __name__ == "__main__":
    run_ingestion()