import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from PIL import Image
import shutil

def analyze_data(train_dir):
    """
    Analyzes the training data to identify class distribution and potential imbalances.
    """
    class_counts = Counter()
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        if os.path.isdir(class_dir):
            num_images = len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])
            class_counts[class_name] = num_images
    return class_counts

def plot_class_distribution(class_counts):
    """
    Plots the distribution of classes in the training data.
    """
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    plt.figure(figsize=(12, 6))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Species')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Species in Training Data')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def calculate_class_weights(class_counts):
    """
    Calculates class weights to address data imbalance.
    """
    total_samples = sum(class_counts.values())
    weights = {class_name: total_samples / (len(class_counts) * count) for class_name, count in class_counts.items()}
    return weights

def balance_dataset(train_dir, target_samples_per_class):
    """
    Balances the dataset by undersampling or oversampling classes.
    """
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        if os.path.isdir(class_dir):
            image_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
            num_images = len(image_files)

            if num_images > target_samples_per_class:
                # Undersample: Randomly remove images
                num_to_remove = num_images - target_samples_per_class
                files_to_remove = np.random.choice(image_files, num_to_remove, replace=False)
                for file_name in files_to_remove:
                    file_path = os.path.join(class_dir, file_name)
                    os.remove(file_path)
                print(f"Undersampled class {class_name} to {target_samples_per_class} samples.")
            elif num_images < target_samples_per_class:
                # Oversample: Duplicate existing images
                num_to_duplicate = target_samples_per_class - num_images
                files_to_duplicate = np.random.choice(image_files, num_to_duplicate, replace=True)
                for file_name in files_to_duplicate:
                    src_path = os.path.join(class_dir, file_name)
                    # Generate a unique filename for the copied image
                    base_name, extension = os.path.splitext(file_name)
                    dst_name = f"{base_name}_copy_{np.random.randint(1000)}.{extension}"
                    dst_path = os.path.join(class_dir, dst_name)
                    copy_image(src_path, dst_path)
                print(f"Oversampled class {class_name} to {target_samples_per_class} samples.")

def copy_image(src, dst):
    """
    Copies an image from source to destination using shutil.copy2 to preserve metadata.
    """
    try:
        shutil.copy2(src, dst)  # Use copy2 to preserve metadata
    except Exception as e:
        print(f"Error copying image: {e}")