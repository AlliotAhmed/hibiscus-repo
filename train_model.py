"""
Hibiscus Leaf Disease Classifier - Model Training Script
========================================================
This script trains a new model using additional data provided by the user.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IMAGE_SIZE = (224, 224)  # Standard size for many pre-trained models
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

def create_model():
    """Create a CNN model for leaf disease classification"""
    # Use MobileNetV2 as the base model (lightweight and works well with TFLite)
    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create the model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')  # Binary classification (healthy vs diseased)
    ])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data(data_dir):
    """Prepare training, validation, and test data generators"""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Only rescaling for validation
    valid_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Create training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )
    
    # Create validation generator
    valid_generator = valid_datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )
    
    return train_generator, valid_generator

def train_model(data_dir, output_dir='models'):
    """Train the model and save it"""
    logger.info(f"Starting model training with data from: {data_dir}")
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    train_generator, valid_generator = prepare_data(data_dir)
    logger.info(f"Class indices: {train_generator.class_indices}")
    
    # Create model
    model = create_model()
    logger.info(f"Model created: {model.summary()}")
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'hibiscus_model_best.h5'),
            save_best_only=True,
            monitor='val_accuracy'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Save the final model
    model_path = os.path.join(output_dir, 'hibiscus_model_final.h5')
    model.save(model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    # Convert to TFLite
    convert_to_tflite(model, output_dir)
    
    return model, history

def convert_to_tflite(model, output_dir):
    """Convert Keras model to TFLite format"""
    logger.info("Converting model to TFLite format...")
    
    # Convert the model to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the TFLite model
    tflite_path = os.path.join(output_dir, 'hibiscus_leaf_classifier.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    logger.info(f"TFLite model saved to: {tflite_path}")
    
    return tflite_path

if __name__ == "__main__":
    # Set up data directory - this should contain 'healthy' and 'diseased' subdirectories
    DATA_DIR = "leaf_dataset"
    
    # Check if the directory exists
    if not os.path.exists(DATA_DIR):
        logger.error(f"Data directory {DATA_DIR} not found!")
        logger.info("Please create a directory with the following structure:")
        logger.info(f"{DATA_DIR}/")
        logger.info(f"├── healthy/  # Healthy leaf images")
        logger.info(f"└── diseased/ # Diseased leaf images")
        exit(1)
    
    # Check if the subdirectories exist
    required_subdirs = ['healthy', 'diseased']
    missing_subdirs = [d for d in required_subdirs if not os.path.exists(os.path.join(DATA_DIR, d))]
    
    if missing_subdirs:
        logger.error(f"Missing subdirectories: {', '.join(missing_subdirs)}")
        logger.info("Please create the following directory structure:")
        logger.info(f"{DATA_DIR}/")
        logger.info(f"├── healthy/  # Healthy leaf images")
        logger.info(f"└── diseased/ # Diseased leaf images")
        exit(1)
    
    # Train the model
    model, history = train_model(DATA_DIR)
    
    logger.info("Model training complete!")