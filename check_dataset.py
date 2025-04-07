"""
Hibiscus Leaf Disease Classifier - Dataset Checker
=================================================
This script analyzes a dataset of leaf images and provides statistics about its contents.
"""

import os
import argparse
from pathlib import Path
import logging
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dataset(dataset_path):
    """
    Check the dataset structure and content
    
    Args:
        dataset_path: Path to the dataset directory
    """
    dataset_path = Path(dataset_path)
    
    # Check if dataset directory exists
    if not dataset_path.exists():
        logger.error(f"Dataset directory does not exist: {dataset_path}")
        return
    
    # Check for class directories
    expected_classes = ['healthy', 'diseased']
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    class_names = [d.name for d in class_dirs]
    
    logger.info(f"Found {len(class_dirs)} class directories: {', '.join(class_names)}")
    
    missing_classes = [c for c in expected_classes if c not in class_names]
    if missing_classes:
        logger.warning(f"Missing expected class directories: {', '.join(missing_classes)}")
    
    # Check image files in each class directory
    stats = {}
    total_images = 0
    
    for class_dir in class_dirs:
        img_extensions = ['.jpg', '.jpeg', '.png']
        
        image_files = []
        for ext in img_extensions:
            image_files.extend(list(class_dir.glob(f"*{ext}")))
            image_files.extend(list(class_dir.glob(f"*{ext.upper()}")))
        
        stats[class_dir.name] = {
            'count': len(image_files),
            'size_stats': {'width': [], 'height': []},
            'invalid_images': []
        }
        
        logger.info(f"Class '{class_dir.name}': {len(image_files)} images")
        total_images += len(image_files)
        
        # Check a sample of images for size and validity
        sample_size = min(len(image_files), 100)  # Check up to 100 images per class
        if sample_size > 0:
            sample_files = np.random.choice(image_files, sample_size, replace=False)
            
            for img_path in sample_files:
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        stats[class_dir.name]['size_stats']['width'].append(width)
                        stats[class_dir.name]['size_stats']['height'].append(height)
                except Exception as e:
                    logger.warning(f"Invalid image file: {img_path} - {str(e)}")
                    stats[class_dir.name]['invalid_images'].append(img_path)
    
    logger.info(f"Total images: {total_images}")
    
    # Calculate size statistics
    for class_name, class_stats in stats.items():
        if class_stats['size_stats']['width']:
            width_mean = np.mean(class_stats['size_stats']['width'])
            height_mean = np.mean(class_stats['size_stats']['height'])
            width_min = np.min(class_stats['size_stats']['width'])
            height_min = np.min(class_stats['size_stats']['height'])
            width_max = np.max(class_stats['size_stats']['width'])
            height_max = np.max(class_stats['size_stats']['height'])
            
            logger.info(f"Class '{class_name}' size statistics:")
            logger.info(f"  Average dimensions: {width_mean:.1f} x {height_mean:.1f}")
            logger.info(f"  Min dimensions: {width_min} x {height_min}")
            logger.info(f"  Max dimensions: {width_max} x {height_max}")
        
        if class_stats['invalid_images']:
            logger.warning(f"Found {len(class_stats['invalid_images'])} invalid images in class '{class_name}'")
    
    return stats

def visualize_dataset(dataset_path, samples_per_class=5):
    """
    Visualize sample images from the dataset
    
    Args:
        dataset_path: Path to the dataset directory
        samples_per_class: Number of sample images to display per class
    """
    dataset_path = Path(dataset_path)
    
    # Get class directories
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    num_classes = len(class_dirs)
    
    if num_classes == 0:
        logger.warning("No class directories found in the dataset")
        return
    
    # Create figure for displaying samples
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(samples_per_class * 3, num_classes * 3))
    if num_classes == 1:
        axes = [axes]  # Ensure axes is a list for consistent indexing
    
    # For each class, display sample images
    for i, class_dir in enumerate(class_dirs):
        # Find image files
        img_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        for ext in img_extensions:
            image_files.extend(list(class_dir.glob(f"*{ext}")))
            image_files.extend(list(class_dir.glob(f"*{ext.upper()}")))
        
        if not image_files:
            logger.warning(f"No images found in class directory: {class_dir}")
            continue
        
        # Select sample images
        sample_size = min(len(image_files), samples_per_class)
        sample_files = np.random.choice(image_files, sample_size, replace=False)
        
        # Display each sample image
        for j, img_path in enumerate(sample_files):
            try:
                img = Image.open(img_path)
                axes[i][j].imshow(img)
                axes[i][j].set_title(f"{class_dir.name}\n{img.size}")
                axes[i][j].axis('off')
            except Exception as e:
                logger.error(f"Error displaying image {img_path}: {str(e)}")
                axes[i][j].text(0.5, 0.5, f"Error: {str(e)}", horizontalalignment='center', verticalalignment='center')
                axes[i][j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_path.parent, 'dataset_samples.png'))
    logger.info(f"Sample visualization saved to: {os.path.join(dataset_path.parent, 'dataset_samples.png')}")

def main():
    parser = argparse.ArgumentParser(description='Check dataset for leaf disease classification.')
    parser.add_argument('--dataset', default='leaf_dataset', help='Path to the dataset directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize sample images from the dataset')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples to visualize per class')
    
    args = parser.parse_args()
    
    # Check dataset structure and content
    stats = check_dataset(args.dataset)
    
    # Visualize dataset samples if requested
    if args.visualize:
        visualize_dataset(args.dataset, args.samples)

if __name__ == "__main__":
    main()