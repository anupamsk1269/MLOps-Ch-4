import sys
import os

# Force the interpreter to look at the standard site-packages first
# This bypasses any local folder naming conflicts
site_pkg = "/usr/local/lib/python3.10/site-packages"
if site_pkg not in sys.path:
    sys.path.insert(0, site_pkg)

# Print diagnostic info to CloudWatch logs
print(f"Python Executable: {sys.executable}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
print(f"Sys Path: {sys.path}")

# Now try the imports
from sagemaker.session import Session
import pandas as pd
#import pandas as pd
import time
#import os
import boto3
import sagemaker
#from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup

def run_ingestion():
    # 1. Setup Sessions
    region = "us-east-1"
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = Session(boto_session=boto_session)
    
    # 2. Identify the Input File
    # SageMaker mounts your S3 bucket data to this local directory in the container
    input_dir = "/opt/ml/processing/input"

    input_file = os.path.join(input_dir, "raw_data.csv")
    
    print(f"📂 Checking for input file at: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file not found! Contents of {input_dir}: {os.listdir(input_dir)}")
        return

    # 3. Load and Prepare Data
    df = pd.read_csv(input_file)
    final_df = pd.DataFrame(processed_records)
    
    # Feature Store requires a unique ID and a Timestamp (EventTime)
    # We create a unique ID for each record based on current time
    current_time = int(time.time())
    df['text_id'] = [f"rec-{current_time}-{i}" for i in range(len(df))]
    df['event_time'] = float(current_time)
    
    # Ensure all data types match your Feature Group (Feature Store is strict)
    df['text'] = df['text'].astype(str)
    
    # 4. Connect to Feature Group
    fg_name = "roberta-sentiment-features"
    feature_group = FeatureGroup(name=fg_name, sagemaker_session=sagemaker_session)
    
    # 5. Ingest
    print(f"🚀 Ingesting {len(df)} records into Feature Group: {fg_name}...")
    # 1. Define only the columns that exist in your Feature Group
    valid_features = ["guid", "event_time", "sentiment", "score"]

    # 2. Filter the dataframe to ONLY these columns
    # This drops the "text", "label", and "text_id" columns that are causing the error
    df_to_ingest = final_df[valid_features]
    try:
        feature_group.ingest(data_frame=df, max_workers=3, wait=True)
        print("✅ Ingestion successfully completed!")
    except ClientError as e:
        print(f"❌ Ingestion failed: {e}")

if __name__ == "__main__":
    run_ingestion()