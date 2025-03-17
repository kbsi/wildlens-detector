# WildLens Detector Dataset

## Dataset Overview

The dataset used for training the wildlife footprint identification model consists of images of various animal footprints. Each image is labeled with the corresponding species name and additional metadata, such as the location and date of capture.

## Dataset Structure

The dataset is organized as follows:

- `Mammifères/`: Main directory containing subdirectories for each animal species.
  - `[Animal Species]/`: Each subdirectory contains footprint images for a specific animal.

This organization allows for easy categorization and training using directory-based image loading techniques such as `ImageDataGenerator` in Keras.

## Image Format

All images are in JPEG format and should be resized to a consistent dimension (e.g., 224x224 pixels) before being fed into the model.

## Data Collection

Images were collected from various sources, including wildlife photography databases, field studies, and contributions from wildlife enthusiasts. Each image has been verified for accuracy in species identification.

## Usage

To use the dataset for training the model:

1. Split the data into training, validation, and test sets using the data preprocessing scripts.
2. Use directory-based image loaders like `tf.keras.preprocessing.image_dataset_from_directory` or `ImageDataGenerator` to load and preprocess images.
3. Refer to `data_preprocessing.py` script for details on preprocessing steps.

Example code to use with this data structure:

```python
# Using ImageDataGenerator with a directory structure
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Training data
train_generator = train_datagen.flow_from_directory(
    'Mammifères/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Validation data
validation_generator = train_datagen.flow_from_directory(
    'Mammifères/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
```

## License

The dataset is provided under the [insert license type here] license. Please ensure compliance with the license terms when using the dataset for research or commercial purposes.
