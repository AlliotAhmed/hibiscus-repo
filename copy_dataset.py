"""
Hibiscus Leaf Disease Classifier - Copy Dataset Script
=====================================================
This script helps copy and organize a dataset of leaf images for training.
"""

import os
import argparse
import shutil
import logging
from pathlib import Path
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def copy_images(source_dir, target_dir, class_name, limit=None):
    """
    Copy images from source directory to target class directory
    
    Args:
        source_dir: Source directory containing images
        target_dir: Target dataset directory (leaf_dataset)
        class_name: Class name ('healthy' or 'diseased')
        limit: Maximum number of images to copy (None for all)
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir) / class_name
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of image files
    img_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    
    for ext in img_extensions:
        image_files.extend(list(source_path.glob(f"*{ext}")))
        image_files.extend(list(source_path.glob(f"*{ext.upper()}")))
    
    if not image_files:
        logger.warning(f"No image files found in {source_path}")
        return 0
    
    # Limit the number of images if specified
    if limit and limit < len(image_files):
        image_files = image_files[:limit]
    
    # Copy each image
    count = 0
    for img_path in image_files:
        try:
            # Validate the image
            try:
                with Image.open(img_path) as img:
                    # Basic validation
                    img.verify()
            except Exception as e:
                logger.warning(f"Skipping invalid image: {img_path} - {str(e)}")
                continue
            
            # Copy the file
            target_file = target_path / img_path.name
            shutil.copy2(img_path, target_file)
            count += 1
            
            if count % 10 == 0:
                logger.info(f"Copied {count} images...")
        except Exception as e:
            logger.error(f"Error copying {img_path}: {str(e)}")
    
    logger.info(f"Successfully copied {count} images to {target_path}")
    return count

def main():
    parser = argparse.ArgumentParser(description='Copy and organize dataset for leaf disease classification.')
    parser.add_argument('--source', required=True, help='Source directory containing images')
    parser.add_argument('--target', default='leaf_dataset', help='Target dataset directory')
    parser.add_argument('--class', dest='class_name', required=True, choices=['healthy', 'diseased'], 
                        help='Class of the images (healthy or diseased)')
    parser.add_argument('--limit', type=int, help='Maximum number of images to copy')
    
    args = parser.parse_args()
    
    # Copy images
    count = copy_images(args.source, args.target, args.class_name, args.limit)
    
    if count > 0:
        logger.info(f"Dataset updated successfully. Added {count} {args.class_name} images.")
    else:
        logger.warning("No images were copied. Please check the source directory.")

if __name__ == "__main__":
    main()