import os
import numpy as np
from PIL import Image
import logging

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """
    Preprocess the uploaded image for the ML model
    
    Args:
        image_path: Path to the uploaded image file
        
    Returns:
        Preprocessed image as numpy array ready for model inference
    """
    try:
        logging.info(f"Preprocessing image: {image_path}")
        
        # Open image
        img = Image.open(image_path)
        
        # Resize to model's expected input dimensions (224x224 is common for many models)
        img = img.resize((224, 224))
        logging.info(f"Resized image to 224x224")
        
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
            logging.info(f"Converted image to RGB mode")
        
        # Convert to numpy array and normalize to [0,1]
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        logging.info(f"Preprocessed image shape: {img_array.shape}, dtype: {img_array.dtype}")
        
        return img_array
    
    except Exception as e:
        logging.error(f"Error preprocessing image: {str(e)}")
        raise Exception(f"Error preprocessing image: {str(e)}")
        
def create_upload_folder():
    """Create upload folder if it doesn't exist"""
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
        logging.info(f"Created upload folder: {upload_folder}")
    return upload_folder
