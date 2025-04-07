import os
import numpy as np
from PIL import Image

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

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
        # Open image
        img = Image.open(image_path)
        
        # Resize to model's expected input dimensions
        img = img.resize((224, 224))
        
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")
