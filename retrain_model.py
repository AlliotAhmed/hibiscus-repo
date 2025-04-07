"""
Hibiscus Leaf Disease Classifier - Model Retraining Module
=========================================================
This script provides a lightweight retraining mechanism for the TFLite model.
"""

import os
import shutil
import random
import uuid
import logging
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
MODEL_PATH = 'attached_assets/hibiscus_leaf_classifier.tflite'
DATASET_DIR = 'leaf_dataset'
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'validation')
HEALTHY_DIR = 'healthy'
DISEASED_DIR = 'diseased'
IMG_HEIGHT = 224
IMG_WIDTH = 224

def ensure_directory_structure():
    """Ensures the dataset directory structure exists"""
    # Create main dataset directories
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    
    # Create class subdirectories for training
    os.makedirs(os.path.join(TRAIN_DIR, HEALTHY_DIR), exist_ok=True)
    os.makedirs(os.path.join(TRAIN_DIR, DISEASED_DIR), exist_ok=True)
    
    # Create class subdirectories for validation
    os.makedirs(os.path.join(VAL_DIR, HEALTHY_DIR), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, DISEASED_DIR), exist_ok=True)


def add_to_dataset(image_path, label):
    """
    Add an image to the training dataset
    
    Args:
        image_path: Path to the image file
        label: Either 'healthy' or 'diseased'
    
    Returns:
        Destination path where the image was saved
    """
    # Ensure directory structure exists
    ensure_directory_structure()
    
    # Validate label
    if label not in [HEALTHY_DIR, DISEASED_DIR]:
        raise ValueError(f"Label must be '{HEALTHY_DIR}' or '{DISEASED_DIR}', got '{label}'")
    
    # Generate a unique filename
    file_ext = os.path.splitext(image_path)[1]
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    
    # Determine if this should go to training or validation
    # 80% training, 20% validation split
    if random.random() < 0.8:
        dest_dir = os.path.join(TRAIN_DIR, label)
    else:
        dest_dir = os.path.join(VAL_DIR, label)
    
    # Copy the file
    dest_path = os.path.join(dest_dir, unique_filename)
    shutil.copy2(image_path, dest_path)
    
    return dest_path


def get_dataset_stats():
    """
    Get statistics about the current dataset
    
    Returns:
        Dictionary with dataset statistics
    """
    # Ensure directory structure exists
    ensure_directory_structure()
    
    try:
        # Count training images
        train_healthy_count = len(os.listdir(os.path.join(TRAIN_DIR, HEALTHY_DIR)))
        train_diseased_count = len(os.listdir(os.path.join(TRAIN_DIR, DISEASED_DIR)))
        
        # Count validation images
        val_healthy_count = len(os.listdir(os.path.join(VAL_DIR, HEALTHY_DIR)))
        val_diseased_count = len(os.listdir(os.path.join(VAL_DIR, DISEASED_DIR)))
        
        # Total counts
        healthy_count = train_healthy_count + val_healthy_count
        diseased_count = train_diseased_count + val_diseased_count
        total_count = healthy_count + diseased_count
        
        # Prepare statistics dictionary
        stats = {
            "train_healthy_count": train_healthy_count,
            "train_diseased_count": train_diseased_count,
            "val_healthy_count": val_healthy_count,
            "val_diseased_count": val_diseased_count,
            "healthy_count": healthy_count,
            "diseased_count": diseased_count,
            "total_count": total_count
        }
        
        return stats
    
    except Exception as e:
        logging.error(f"Error getting dataset statistics: {str(e)}")
        return {
            "train_healthy_count": 0,
            "train_diseased_count": 0,
            "val_healthy_count": 0,
            "val_diseased_count": 0,
            "healthy_count": 0,
            "diseased_count": 0,
            "total_count": 0
        }


def create_lightweight_model():
    """Create a lightweight CNN model for leaf disease classification"""
    # Create a sequential model
    model = keras.Sequential([
        # Base model
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification: healthy vs diseased
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def prepare_data():
    """Prepare training and validation data generators"""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        class_mode='binary'
    )
    
    # Load validation data
    validation_generator = validation_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        class_mode='binary'
    )
    
    return train_generator, validation_generator


def retrain_model():
    """
    Retrain the model with the current dataset
    
    Returns:
        Dictionary with training results
    """
    # Get dataset statistics
    stats = get_dataset_stats()
    
    # Check if we have enough data
    if stats["total_count"] < 4 or stats["healthy_count"] == 0 or stats["diseased_count"] == 0:
        return {
            "success": False,
            "message": "Not enough training data. Need at least 1 image of each class."
        }
    
    try:
        # Create the model
        model = create_lightweight_model()
        
        # Prepare data generators
        train_generator, validation_generator = prepare_data()
        
        # Create early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        )
        
        # Train the model
        start_time = time.time()
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=10,
            callbacks=[early_stopping]
        )
        training_time = time.time() - start_time
        
        # Get the training and validation accuracy
        train_accuracy = max(history.history['accuracy']) * 100
        val_accuracy = max(history.history['val_accuracy']) * 100
        
        # Convert to TFLite format
        tflite_model = convert_to_tflite(model)
        
        # Save the TFLite model
        with open(MODEL_PATH, 'wb') as f:
            f.write(tflite_model)
        
        return {
            "success": True,
            "message": "Model trained and saved successfully.",
            "train_accuracy": round(train_accuracy, 2),
            "val_accuracy": round(val_accuracy, 2),
            "training_time": round(training_time, 2)
        }
    
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        return {
            "success": False,
            "message": f"Training error: {str(e)}"
        }


def convert_to_tflite(model):
    """Convert Keras model to TFLite format"""
    # Convert the model to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    return tflite_model