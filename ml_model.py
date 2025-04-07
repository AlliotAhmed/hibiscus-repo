import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Dictionary for disease classes
DISEASE_CLASSES = {
    0: "Healthy",
    1: "Diseased"
}

# Dictionary for disease information
DISEASE_INFO = {
    "Healthy": "This hibiscus leaf appears healthy with no signs of disease.",
    "Diseased": "This hibiscus leaf shows signs of disease. Common hibiscus diseases include powdery mildew, leaf spot, aphids infestation, and hibiscus chlorotic ringspot virus."
}

# Initialize model as None, will be loaded on first prediction
model = None

def create_model():
    """
    Create a simple CNN model for binary classification of hibiscus leaves
    This is a placeholder model - in a real application, you'd use a pre-trained model
    """
    model = Sequential([
        # Use MobileNetV2 as base model with pretrained weights
        tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        ),
        # Add classification head
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # In a real application, we would load pre-trained weights here
    # model.load_weights('path_to_weights.h5')
    
    return model

def load_model():
    """Load or create the model"""
    global model
    if model is None:
        logging.info("Creating new model instance")
        model = create_model()
    return model

def predict_disease(image):
    """
    Make prediction on the preprocessed image
    
    Args:
        image: Preprocessed image as numpy array
        
    Returns:
        Tuple of (prediction_label, confidence_score)
    """
    try:
        # Load the model
        model = load_model()
        
        # For demonstration purposes, we're simulating a prediction here
        # In a real-world application, you'd use a properly trained model
        
        # Apply MobileNetV2 preprocessing
        preprocessed_img = preprocess_input(image * 255.0)
        
        # Get prediction
        prediction = model.predict(preprocessed_img)
        confidence = float(prediction[0][0])
        
        # Threshold the prediction (0.5 is the standard threshold for binary classification)
        predicted_class = 1 if confidence >= 0.5 else 0
        
        # Calculate confidence percentage
        if predicted_class == 1:
            confidence_percentage = confidence * 100
        else:
            confidence_percentage = (1 - confidence) * 100
        
        prediction_label = DISEASE_CLASSES[predicted_class]
        prediction_info = DISEASE_INFO[prediction_label]
        
        return {
            "label": prediction_label,
            "confidence": round(confidence_percentage, 2),
            "info": prediction_info
        }, confidence_percentage
        
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        raise Exception(f"Error making prediction: {str(e)}")
