import sys
import os

# This forces the script to look inside your virtual environment's library folder
venv_path = os.path.join(os.getcwd(), '.venv/lib/python3.12/site-packages')
if venv_path not in sys.path:
    sys.path.insert(0, venv_path)
print("✅ Set the vertual envionment path")
import boto3
import sagemaker
from sagemaker.session import Session
#from sagemaker import Session
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum
import time

# 1. Configuration - PASTE YOUR ARN HERE
ROLE_ARN = "arn:aws:iam::372362723027:role/AmazonSageMaker-ExecutionRole-1969"
REGION = "us-east-1"
FEATURE_GROUP_NAME = "roberta-sentiment-features"

# 2. Setup Boto3 and SageMaker Sessions
boto_session = boto3.Session(region_name=REGION)
sagemaker_client = boto_session.client(service_name='sagemaker')
featurestore_runtime = boto_session.client(service_name='sagemaker-featurestore-runtime')

sagemaker_session = Session(
    boto_session=boto_session,
    sagemaker_client=sagemaker_client,
    sagemaker_featurestore_runtime_client=featurestore_runtime
)

# 3. Define the Schema (Features for RoBERTa)
feature_definitions = [
    FeatureDefinition(feature_name="text_id", feature_type=FeatureTypeEnum.STRING),
    FeatureDefinition(feature_name="sentiment_label", feature_type=FeatureTypeEnum.STRING),
    FeatureDefinition(feature_name="sentiment_score", feature_type=FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition(feature_name="event_time", feature_type=FeatureTypeEnum.FRACTIONAL)
]

# 4. Initialize and Create the Feature Group
sentiment_fg = FeatureGroup(name=FEATURE_GROUP_NAME, sagemaker_session=sagemaker_session)
sentiment_fg.feature_definitions = feature_definitions

default_bucket = sagemaker_session.default_bucket()
print("Default bucket name is {default_bucket}")
print(f"🚀 Creating Feature Group: {FEATURE_GROUP_NAME} in bucket {default_bucket}...")

# 5. Create!
offline_s3_uri = "s3://sagemaker-us-east-1-372362723027/offline-store"

print(f"🚀 Creating Feature Group: {FEATURE_GROUP_NAME}...")
sentiment_fg.create(
    s3_uri=offline_s3_uri,  # Using the hardcoded URI
    record_identifier_name="text_id",
    event_time_feature_name="event_time",
    role_arn=ROLE_ARN,
    enable_online_store=True
)

# 5. Wait for the status to become 'Created'
def wait_for_fg():
    status = sentiment_fg.describe().get("FeatureGroupStatus")
    while status == "Creating":
        print("Waiting for creation...")
        time.sleep(10)
        status = sentiment_fg.describe().get("FeatureGroupStatus")
    print(f"✅ Success! Feature Group Status: {status}")

wait_for_fg()