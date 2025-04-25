import os
from google.cloud import storage
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def download_model_from_gcs():
    """
    Download the model file from Google Cloud Storage to a temporary location
    """
    try:
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        model_path = os.path.join(temp_dir, 'hibiscus_leaf_classifier.tflite')
        
        # Check if model already exists in temp
        if os.path.exists(model_path):
            logging.info("Model already downloaded to temp directory")
            return model_path
            
        # Initialize GCS client
        storage_client = storage.Client()
        
        # Get the bucket
        bucket_name = os.environ.get('GCS_BUCKET_NAME')
        if not bucket_name:
            logging.error("GCS_BUCKET_NAME environment variable not set")
            return None
            
        bucket = storage_client.bucket(bucket_name)
        
        # Get the blob
        blob = bucket.blob('models/hibiscus_leaf_classifier.tflite')
        
        # Download to temporary file
        blob.download_to_filename(model_path)
        
        logging.info(f"Model downloaded to {model_path}")
        return model_path
        
    except Exception as e:
        logging.error(f"Error downloading model from GCS: {str(e)}")
        return None 