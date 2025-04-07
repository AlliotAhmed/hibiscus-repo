"""
Hibiscus Leaf Disease Classifier - Data Preparation Script
=========================================================
This script helps prepare and organize new datasets for training.
"""

import os
import shutil
import argparse
import logging
from pathlib import Path
import random
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dataset_structure(dataset_path, clean=False):
    """Create the dataset directory structure"""
    dataset_path = Path(dataset_path)
    
    if clean and dataset_path.exists():
        logger.info(f"Cleaning existing dataset directory: {dataset_path}")
        shutil.rmtree(dataset_path)
    
    # Create main dataset directory
    dataset_path.mkdir(exist_ok=True)
    
    # Create class subdirectories
    classes = ['healthy', 'diseased']
    for cls in classes:
        class_dir = dataset_path / cls
        class_dir.mkdir(exist_ok=True)
        logger.info(f"Created directory: {class_dir}")
    
    return dataset_path

def organize_from_directory(source_dir, dataset_path, split_ratio=0.8):
    """
    Organize images from a directory into the dataset structure
    
    Args:
        source_dir: Directory containing images to organize
        dataset_path: Root directory of the dataset
        split_ratio: Train/validation split ratio (0.8 = 80% training, 20% validation)
    """
    source_dir = Path(source_dir)
    dataset_path = Path(dataset_path)
    
    # Check if source directory exists
    if not source_dir.exists() or not source_dir.is_dir():
        logger.error(f"Source directory does not exist: {source_dir}")
        return False
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(source_dir.glob(f"*{ext}")))
        image_files.extend(list(source_dir.glob(f"*{ext.upper()}")))
    
    if not image_files:
        logger.error(f"No image files found in: {source_dir}")
        return False
    
    logger.info(f"Found {len(image_files)} images in {source_dir}")
    
    # Get class label from directory name
    class_name = source_dir.name.lower()
    if class_name not in ['healthy', 'diseased']:
        logger.warning(f"Directory name '{class_name}' doesn't match expected classes (healthy/diseased)")
        class_input = input(f"Specify class for images in {source_dir} (healthy/diseased): ").strip().lower()
        if class_input in ['healthy', 'diseased']:
            class_name = class_input
        else:
            logger.error("Invalid class specified. Skipping directory.")
            return False
    
    # Copy images to dataset directory
    target_dir = dataset_path / class_name
    
    # Create target directory if it doesn't exist
    if not target_dir.exists():
        target_dir.mkdir(parents=True)
    
    # Copy each image file
    for img_path in image_files:
        try:
            # Validate image file
            try:
                with Image.open(img_path) as img:
                    # Basic validation - make sure the file is a valid image
                    img.verify()
            except Exception as e:
                logger.warning(f"Skipping invalid image: {img_path} - {str(e)}")
                continue
            
            # Copy the file
            target_path = target_dir / img_path.name
            shutil.copy2(img_path, target_path)
            logger.debug(f"Copied: {img_path} -> {target_path}")
        except Exception as e:
            logger.error(f"Error copying {img_path}: {str(e)}")
    
    logger.info(f"Organized images into: {target_dir}")
    return True

def organize_dataset(source_dirs, dataset_path):
    """
    Organize multiple source directories into a dataset structure
    
    Args:
        source_dirs: List of source directories to organize
        dataset_path: Root directory of the dataset
    """
    dataset_path = Path(dataset_path)
    
    # Create dataset structure
    create_dataset_structure(dataset_path)
    
    # Organize each source directory
    for source_dir in source_dirs:
        organize_from_directory(source_dir, dataset_path)
    
    # Count images in each class
    for class_name in ['healthy', 'diseased']:
        class_dir = dataset_path / class_name
        if class_dir.exists():
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
            logger.info(f"Class '{class_name}': {len(image_files)} images")

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for hibiscus leaf disease classification.')
    parser.add_argument('--source', nargs='+', help='Source directories containing images')
    parser.add_argument('--output', default='leaf_dataset', help='Output dataset directory')
    parser.add_argument('--clean', action='store_true', help='Clean existing dataset directory')
    
    args = parser.parse_args()
    
    if args.source:
        organize_dataset(args.source, args.output)
    else:
        # Interactive mode
        logger.info("Interactive mode - preparing dataset structure")
        dataset_path = create_dataset_structure(args.output, args.clean)
        
        logger.info(f"Dataset directory created: {dataset_path}")
        logger.info("Please copy your images into the 'healthy' and 'diseased' subdirectories")
        logger.info("After copying the images, you can train the model using: python train_model.py")

if __name__ == "__main__":
    main()