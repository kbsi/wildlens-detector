# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from tqdm import tqdm  # Also needed for the preprocess_and_save_dataset function
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, MobileNetV3Large, ResNet50V2, DenseNet121
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, Average, Input, Activation, Multiply
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Model
import cv2
from tensorflow.keras.callbacks import LearningRateScheduler


def add_subtle_noise(image):
    """Ajoute un bruit léger pour simuler les variations de texture du sol"""
    if np.random.random() > 0.7:  # Seulement 30% du temps
        if isinstance(image, tf.Tensor):
            image_np = image.numpy()
        else:
            image_np = image

        # Bruit gaussien très léger
        noise = np.random.normal(0, 0.02, image_np.shape)
        image_np = np.clip(image_np + noise, 0, 1)

        if isinstance(image, tf.Tensor):
            return tf.convert_to_tensor(image_np, dtype=tf.float32)
        return image_np

    return image


def create_train_val_test_dirs(data, temp_dir, train_size=0.7, val_size=0.15, test_size=0.15):
    """
    Crée une structure de dossiers pour l'entraînement, la validation et le test.

    Args:
        data (pd.DataFrame): DataFrame contenant les colonnes 'image_path' et 'species'
        temp_dir (str): Chemin du dossier temporaire où créer les sous-dossiers
        train_size, val_size, test_size (float): Proportions des données pour chaque ensemble

    Returns:
        tuple: Chemins des dossiers d'entraînement, validation et test
    """

    # Réinitialiser les dossiers existants
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    # Créer les dossiers principaux
    train_dir = os.path.join(temp_dir, 'train')
    val_dir = os.path.join(temp_dir, 'validation')
    test_dir = os.path.join(temp_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Obtenir la liste des espèces uniques
    species_list = data['species'].unique()

    # Créer un sous-dossier pour chaque espèce dans train, val et test
    for species in species_list:
        os.makedirs(os.path.join(train_dir, species), exist_ok=True)
        os.makedirs(os.path.join(val_dir, species), exist_ok=True)
        os.makedirs(os.path.join(test_dir, species), exist_ok=True)

    # Pour chaque espèce, diviser les données
    for species in species_list:
        species_data = data[data['species'] == species]

        # Première division: séparer test et non-test
        non_test_data, test_data = train_test_split(
            species_data,
            test_size=test_size,
            random_state=42
        )

        # Deuxième division: séparer train et validation à partir des données non-test
        train_data, val_data = train_test_split(
            non_test_data,
            test_size=val_size/(train_size+val_size),
            random_state=42
        )

        # Copier les images dans les dossiers appropriés
        base_img_dir = 'data/Mammiferes_augmented/'

        # Copier les images d'entraînement
        for _, row in train_data.iterrows():
            src_path = os.path.join(base_img_dir, row['image_path'])
            dst_path = os.path.join(
                train_dir, species, os.path.basename(row['image_path']))
            safe_copy(src_path, dst_path)

        # Copier les images de validation
        for _, row in val_data.iterrows():
            src_path = os.path.join(base_img_dir, row['image_path'])
            dst_path = os.path.join(
                val_dir, species, os.path.basename(row['image_path']))
            safe_copy(src_path, dst_path)

        # Copier les images de test
        for _, row in test_data.iterrows():
            src_path = os.path.join(base_img_dir, row['image_path'])
            dst_path = os.path.join(
                test_dir, species, os.path.basename(row['image_path']))
            safe_copy(src_path, dst_path)

    # Compter et afficher les statistiques
    train_counts = {}
    val_counts = {}
    test_counts = {}

    for species in species_list:
        train_counts[species] = len(
            os.listdir(os.path.join(train_dir, species)))
        val_counts[species] = len(os.listdir(os.path.join(val_dir, species)))
        test_counts[species] = len(os.listdir(os.path.join(test_dir, species)))

    print("Répartition des données:")
    print(f"  Entraînement: {sum(train_counts.values())} images")
    print(f"  Validation: {sum(val_counts.values())} images")
    print(f"  Test: {sum(test_counts.values())} images")

    return train_dir, val_dir, test_dir


def plot_confusion_matrix(model, generator):
    # Prédire les classes
    y_pred_probs = model.predict(generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = generator.classes[:len(y_pred)]

    # Convertir les indices en noms de classes
    idx_to_class = {v: k for k, v in generator.class_indices.items()}
    class_names = [idx_to_class[i]
                   for i in range(len(generator.class_indices))]

    # Calculer et normaliser la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Visualisation
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Classe réelle')
    plt.xlabel('Classe prédite')
    plt.title('Matrice de confusion normalisée')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Calcul de métriques par classe
    class_accuracy = np.diag(cm) / cm.sum(axis=1)
    for i, (name, acc) in enumerate(zip(class_names, class_accuracy)):
        print(f"{name}: {acc:.2f} ({cm[i].sum()} échantillons)")

    return cm, class_names


def preprocess_footprint_image(img_path):
    # Code existant pour charger l'image
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Amélioration du contraste via CLAHE avec paramètres ajustés
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. Débruitage bilatéral (préserve mieux les contours)
    denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    # 3. Détection des contours avec Canny
    edges = cv2.Canny(denoised, 50, 150)

    # 4. Fermeture morphologique pour connecter les contours proches
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. Trouver et remplir les contours principaux uniquement
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Garder uniquement les 5 plus grands contours
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Créer un masque pour les contours principaux
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, -1)

    # Appliquer le masque à l'image originale
    result = cv2.bitwise_and(gray, mask)

    # Normaliser et redimensionner
    result = cv2.resize(result, (224, 224))
    result = result.astype(np.float32) / 255.0

    # Convertir en RGB
    result_rgb = np.stack([result, result, result], axis=-1)

    return result_rgb

# Prétraiter toutes les images et les sauvegarder


def preprocess_and_save_dataset(data, output_dir):
    """Prétraite toutes les images et les sauvegarde"""
    os.makedirs(output_dir, exist_ok=True)

    for species in data['species'].unique():
        species_dir = os.path.join(output_dir, species)
        os.makedirs(species_dir, exist_ok=True)

    for _, row in tqdm(data.iterrows(), total=len(data)):
        species = row['species']
        img_path = os.path.join(
            'data/Mammiferes_augmented/', row['image_path'])

        # Prétraiter l'image
        processed_img = preprocess_footprint_image(img_path)

        if processed_img is not None:
            # Sauvegarder l'image prétraitée
            output_path = os.path.join(
                output_dir, species, os.path.basename(row['image_path']))
            cv2.imwrite(output_path, (processed_img * 255).astype(np.uint8))

# Créer un modèle d'ensemble pour améliorer les performances


def create_ensemble_model(num_classes, input_shape=(224, 224, 3)):
    # Entrée commune
    input_tensor = Input(shape=input_shape)

    # Modèle 1: ResNet50V2
    base1 = ResNet50V2(include_top=False, weights='imagenet',
                       input_tensor=input_tensor)
    x1 = GlobalAveragePooling2D()(base1.output)
    x1 = Dense(128, activation='relu')(x1)
    x1 = Dropout(0.5)(x1)
    output1 = Dense(num_classes, activation='softmax', name='output1')(x1)

    # Modèle 2: EfficientNetB0
    base2 = EfficientNetB0(
        include_top=False, weights='imagenet', input_tensor=input_tensor)
    x2 = GlobalAveragePooling2D()(base2.output)
    x2 = Dense(128, activation='relu')(x2)
    x2 = Dropout(0.5)(x2)
    output2 = Dense(num_classes, activation='softmax', name='output2')(x2)

    # Modèle 3: DenseNet121
    base3 = DenseNet121(include_top=False, weights='imagenet',
                        input_tensor=input_tensor)
    x3 = GlobalAveragePooling2D()(base3.output)
    x3 = Dense(128, activation='relu')(x3)
    x3 = Dropout(0.5)(x3)
    output3 = Dense(num_classes, activation='softmax', name='output3')(x3)

    # Combiner les prédictions
    outputs = Average()([output1, output2, output3])

    # Modèle complet
    ensemble = Model(inputs=input_tensor, outputs=outputs)

    # Geler tous les modèles de base
    for layer in base1.layers:
        layer.trainable = False
    for layer in base2.layers:
        layer.trainable = False
    for layer in base3.layers:
        layer.trainable = False

    return ensemble

# Fonction pour ajouter du bruit aléatoire simulant différentes textures de sol


def add_random_noise(image):
    if np.random.random() > 0.5:  # 50% de chance d'appliquer du bruit
        # Convert TensorFlow tensor to NumPy array if necessary
        if isinstance(image, tf.Tensor):
            image_np = image.numpy()
        else:
            image_np = image

        noise_type = np.random.choice(['gaussian', 'speckle', 'salt_pepper'])

        if noise_type == 'gaussian':
            row, col, ch = image_np.shape
            mean = 0
            sigma = np.random.uniform(0.01, 0.05)
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            noisy = image_np + gauss
            result = np.clip(noisy, 0, 1)

        elif noise_type == 'speckle':
            row, col, ch = image_np.shape
            speckle = np.random.randn(row, col, ch) * \
                np.random.uniform(0.05, 0.1)
            noisy = image_np + image_np * speckle
            result = np.clip(noisy, 0, 1)

        else:  # salt_pepper
            s_vs_p = 0.5
            amount = np.random.uniform(0.01, 0.05)
            noisy = np.copy(image_np)

            # Calculate total number of pixels
            total_pixels = image_np.shape[0] * \
                image_np.shape[1] * image_np.shape[2]

            # Salt
            num_salt = np.ceil(amount * total_pixels * s_vs_p)
            # Get random coordinates
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image_np.shape]
            noisy[coords[0], coords[1], :] = 1

            # Pepper
            num_pepper = np.ceil(amount * total_pixels * (1 - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image_np.shape]
            noisy[coords[0], coords[1], :] = 0

            result = noisy

        # Convert back to tensor if the input was a tensor
        if isinstance(image, tf.Tensor):
            return tf.convert_to_tensor(result, dtype=tf.float32)
        return result

    return image

# Add this function to randomly resize images during training


def multi_scale_augmentation(image):
    """Randomly resize the image to different scales during training"""
    height, width = image.shape[0], image.shape[1]

    # Define several scales (0.8x to 1.2x)
    scale = np.random.uniform(0.8, 1.2)
    new_height = int(height * scale)
    new_width = int(width * scale)

    # Resize the image
    resized = tf.image.resize(image, [new_height, new_width])

    # Pad or crop to original size
    resized = tf.image.resize_with_crop_or_pad(resized, height, width)

    return resized

# Fonction pour combiner les prédictions des modèles


def ensemble_predictions(models, data_generator):
    all_preds = []
    for model in models:
        preds = model.predict(data_generator)
        all_preds.append(preds)

    # Moyenne des prédictions
    avg_preds = np.mean(all_preds, axis=0)
    return avg_preds


def extract_footprint_features(image_path):
    """Extrait des caractéristiques morphologiques des empreintes."""
    try:
        # Charger l'image
        img = cv2.imread(image_path)
        if img is None:
            # Retourner des zéros si l'image ne peut être chargée
            return np.zeros(25)

        # Convertir en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Appliquer un seuil adaptatif pour isoler l'empreinte du fond
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Trouver les contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculer des caractéristiques morphologiques
        features = []
        if contours:
            # Prendre les 5 plus grands contours
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

            for c in contours:
                area = cv2.contourArea(c)
                perimeter = cv2.arcLength(c, True)

                # Circularité
                circularity = 4 * np.pi * area / \
                    (perimeter * perimeter) if perimeter > 0 else 0

                # Rectangle englobant
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / h if h > 0 else 0

                # Convexité
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                convexity = area / hull_area if hull_area > 0 else 0

                features.extend(
                    [area, perimeter, circularity, aspect_ratio, convexity])

        # Compléter avec des zéros si nécessaire
        while len(features) < 25:
            features.append(0)

        return np.array(features)

    except Exception as e:
        print(f"Erreur lors de l'extraction des caractéristiques: {e}")
        return np.zeros(25)


def visualize_model_predictions(model, data_generator, num_samples=10):
    """Visualise les prédictions du modèle sur quelques exemples."""
    # Réinitialiser le générateur
    data_generator.reset()

    # Obtenir un batch d'images et de labels
    images, labels = next(data_generator)

    # Faire des prédictions
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)

    # Convertir les indices en noms de classes
    class_indices = data_generator.class_indices
    idx_to_class = {v: k for k, v in class_indices.items()}

    # Visualiser
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i in range(min(num_samples, len(images))):
        # Afficher l'image
        axes[i].imshow(images[i])

        # Vraie classe et prédiction
        true_class = idx_to_class[true_classes[i]]
        pred_class = idx_to_class[predicted_classes[i]]
        confidence = predictions[i, predicted_classes[i]] * 100

        title = f"Vraie: {true_class}\nPred: {pred_class}\nConf: {confidence:.1f}%"

        # Colorer le titre en fonction de la prédiction
        color = 'green' if true_classes[i] == predicted_classes[i] else 'red'
        axes[i].set_title(title, color=color)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def analyze_confusion_matrix(model, data_generator):
    """Analyse détaillée de la matrice de confusion."""
    # Réinitialiser le générateur
    data_generator.reset()

    # Prédictions
    y_pred = model.predict(data_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Vraies classes
    y_true = data_generator.classes[:len(y_pred_classes)]

    # Convertir les indices en noms de classes
    class_indices = data_generator.class_indices
    idx_to_class = {v: k for k, v in class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(class_indices))]

    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred_classes)

    # Normaliser la matrice
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Tracer la matrice
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.title('Matrice de confusion normalisée')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Identifier les paires de classes les plus confondues
    confusion_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append(
                    (class_names[i], class_names[j], cm[i, j], cm_normalized[i, j]))

    # Trier par nombre d'erreurs
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)

    # Afficher les 10 paires les plus problématiques
    print("Top 10 des confusions entre classes:")
    for true_class, pred_class, count, ratio in confusion_pairs[:10]:
        print(
            f"  {true_class} classifié comme {pred_class}: {count} fois ({ratio*100:.1f}%)")


def filter_classes_with_few_samples(data, min_samples_threshold=30):
    """
    Filtre les classes avec un nombre d'échantillons inférieur à un seuil donné.

    Args:
        data (pd.DataFrame): DataFrame contenant les colonnes 'image_path' et 'species'
        min_samples_threshold (int): Seuil minimum d'échantillons par classe

    Returns:
        pd.DataFrame: DataFrame filtré
    """
    class_counts = data['species'].value_counts()
    classes_to_keep = class_counts[class_counts >= min_samples_threshold].index
    filtered_data = data[data['species'].isin(classes_to_keep)]
    return filtered_data


def train_model():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Load dataset
    data = pd.read_csv('data/footprint_dataset_augmented.csv')

    # Create temporary directory for dataset
    TEMP_DIR = 'data/temp_dataset'

    # First, create initial directories to analyze class distribution
    train_dir, val_dir, test_dir = create_train_val_test_dirs(data, TEMP_DIR)

    # Set up data generator for analysis only
    temp_datagen = ImageDataGenerator(rescale=1./255)
    temp_generator = temp_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )

    # Analyze class distribution
    class_counts = {}
    for class_name, class_idx in temp_generator.class_indices.items():
        class_counts[class_name] = np.sum(temp_generator.classes == class_idx)

    print("Distribution des classes dans l'ensemble d'entraînement:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1]):
        print(f"  {class_name}: {count} échantillons")

    # Identify problematic classes
    small_classes = [class_name for class_name,
                     count in class_counts.items() if count < 20]
    print(f"\nClasses avec très peu d'échantillons ({len(small_classes)}):")
    for class_name in small_classes:
        print(f"  {class_name}")

    # Filter classes with too few samples
    min_samples_threshold = 30  # Seuil minimum d'échantillons par classe
    filtered_data = filter_classes_with_few_samples(
        data, min_samples_threshold)

    # Définir la liste des classes conservées
    classes_to_keep = filtered_data['species'].unique().tolist()
    print(
        f"Réduction du nombre de classes de {len(data['species'].unique())} à {len(classes_to_keep)}")
    print(f"Classes conservées: {classes_to_keep}")

    # IMPORTANT: Clean up the temporary directory before recreating it with filtered data
    import shutil
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    # Recreate directories with only the filtered classes
    train_dir, val_dir, test_dir = create_train_val_test_dirs(
        filtered_data, TEMP_DIR)

    # Verify that only the expected classes are present
    actual_classes = os.listdir(train_dir)
    unexpected_classes = [
        cls for cls in actual_classes if cls not in classes_to_keep]
    if unexpected_classes:
        print(
            f"WARNING: Found unexpected classes in training directory: {unexpected_classes}")

    # Add this after filtering data but before creating train/val/test directories
    print("Preprocessing images...")
    PROCESSED_DIR = 'data/preprocessed_footprints'
    preprocess_and_save_dataset(filtered_data, PROCESSED_DIR)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,  # Rotations plus limitées
        width_shift_range=0.1,  # Déplacements moins extrêmes
        height_shift_range=0.1,
        zoom_range=[0.9, 1.1],  # Zoom moins extrême
        brightness_range=[0.8, 1.2],
        fill_mode='constant',
        cval=0.0,  # Fond noir pour mieux voir les contours
        horizontal_flip=True,
        vertical_flip=False,  # Les empreintes ont souvent une orientation

        # Supprimer les augmentations trop agressives
        preprocessing_function=lambda img: add_subtle_noise(img)
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # Create the filtered generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    # Nombre de classes après filtrage
    num_classes = len(classes_to_keep)

    # Calculate class weights for the FILTERED data
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )

    # Convert to dictionary for Keras
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    print("Poids des classes après filtrage:")
    for class_idx, weight in class_weight_dict.items():
        class_name = list(train_generator.class_indices.keys())[
            list(train_generator.class_indices.values()).index(class_idx)]
        print(f"  {class_name}: {weight:.2f}")

    # MODIFY THIS SECTION: Use a simple EfficientNetB0 model instead of fusion model
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freezer le modèle de base
    base_model.trainable = False

    # Construire un modèle séquentiel (évite les problèmes de compatibilité)
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10,
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-6),
        ModelCheckpoint(
            '../models/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    history_phase1 = model.fit(
        train_generator,
        epochs=30,
        validation_data=validation_generator,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Phase 2: Fine-tuning avec un taux d'apprentissage très faible
    base_model.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_phase2 = model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
        initial_epoch=len(history_phase1.history['loss'])
    )

    # Phase 3: Unfreeze the last 30% of the base model
    print("Phase 3: Fine-tuning the last 30% of layers...")
    base_model.trainable = True

    # Calculate how many layers to unfreeze (30% of total)
    total_layers = len(base_model.layers)
    layers_to_unfreeze = int(0.3 * total_layers)

    # Freeze all layers except the last X%
    for layer in base_model.layers[:-layers_to_unfreeze]:
        layer.trainable = False

    # Show how many layers are being trained
    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    print(f"Training {trainable_count} of {total_layers} layers in base model")

    model.compile(
        optimizer=Adam(learning_rate=5e-5),  # Slightly higher learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_phase3 = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=15,  # Shorter phase
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
        initial_epoch=len(history_phase2.history['loss'])
    )

    # Phase 4: Fine-tuning all layers with very small learning rate
    print("Phase 4: Fine-tuning all layers...")
    # Unfreeze all layers
    for layer in base_model.layers:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=1e-6),  # Very small learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Fix the initial_epoch calculation to safely handle missing keys
    initial_epoch = len(history_phase1.history.get('loss', []))
    if hasattr(history_phase2, 'history') and 'loss' in history_phase2.history:
        initial_epoch += len(history_phase2.history['loss'])

    history_phase4 = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=10,  # Short final phase
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
        initial_epoch=initial_epoch
    )

    # Entraîner trois modèles différents
    models = []

    # Modèle 1: EfficientNetB0
    # (Déjà défini et entraîné ci-dessus)
    models.append(model)

    # Modèle 2: MobileNetV3
    base_model2 = MobileNetV3Large(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        alpha=1.0
    )
    base_model2.trainable = False

    model2 = Sequential([
        base_model2,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])

    model2.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Entraîner le modèle 2
    history_model2 = model2.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=20,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    models.append(model2)

    # Modèle 3: ResNet50V2
    base_model3 = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model3.trainable = False

    model3 = Sequential([
        base_model3,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])

    model3.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Entraîner le modèle 3
    history_model3 = model3.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=20,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    models.append(model3)

    # Save all models
    model.save('../models/model1_mobilenet.keras')
    model2.save('../models/model2_mobilenetv3.keras')
    model3.save('../models/model3_resnet.keras')

    # Obtenir les prédictions d'ensemble
    ensemble_preds, weights = weighted_ensemble_predictions(
        models, validation_generator, test_generator)
    ensemble_classes = np.argmax(ensemble_preds, axis=1)

    # Visualiser quelques prédictions
    visualize_model_predictions(model, test_generator)

    # Analyser les confusions
    analyze_confusion_matrix(model, test_generator)

    print("\n--- Évaluation standard ---")
    for i, model_name in enumerate(['EfficientNetB0', 'MobileNetV3Large', 'ResNet50V2']):
        if i < len(models):
            loss, acc = models[i].evaluate(test_generator)
            print(
                f"Modèle {model_name}: Accuracy = {acc:.4f}, Loss = {loss:.4f}")

    print("\n--- Évaluation avec Test-Time Augmentation ---")
    for i, model_name in enumerate(['EfficientNetB0', 'MobileNetV3Large', 'ResNet50V2']):
        if i < len(models):
            all_preds, all_labels = evaluate_with_tta(
                models[i], test_generator)

            # Convertir les prédictions en classes
            pred_classes = np.argmax(all_preds, axis=1)
            true_classes = np.argmax(all_labels, axis=1)

            # Calculer l'accuracy
            tta_accuracy = np.mean(pred_classes == true_classes)
            print(
                f"Modèle {model_name} avec TTA: Accuracy = {tta_accuracy:.4f}")

    # Évaluer l'ensemble avec TTA
    print("\n--- Évaluation de l'ensemble avec TTA ---")
    ensemble_preds_tta = []

    # Pour chaque image dans le générateur de test
    for i in range(len(test_generator)):
        batch_x, _ = test_generator[i]
        batch_ensemble_preds = []

        # Pour chaque image dans le batch
        for j in range(len(batch_x)):
            img = batch_x[j]
            # Collecter les prédictions avec TTA pour chaque modèle
            model_preds = []
            for model in models:
                tta_pred = test_time_augmentation(model, img)
                model_preds.append(tta_pred)

            # Moyenne pondérée des prédictions
            weighted_pred = np.zeros_like(model_preds[0])
            for k, pred in enumerate(model_preds):
                # weights est déjà calculé dans weighted_ensemble_predictions
                weighted_pred += pred * weights[k]

            batch_ensemble_preds.append(weighted_pred)

        ensemble_preds_tta.extend(batch_ensemble_preds)

    # Convertir en array numpy
    ensemble_preds_tta = np.array(ensemble_preds_tta)

    # Comparer avec les vraies étiquettes
    ensemble_classes_tta = np.argmax(ensemble_preds_tta, axis=1)
    true_classes = np.array([])

    # Récupérer les vraies classes
    for i in range(len(test_generator)):
        _, batch_y = test_generator[i]
        true_batch_classes = np.argmax(batch_y, axis=1)
        true_classes = np.append(true_classes, true_batch_classes)

    # Calculer l'accuracy
    ensemble_tta_accuracy = np.mean(
        ensemble_classes_tta == true_classes[:len(ensemble_classes_tta)])
    print(f"Ensemble avec TTA: Accuracy = {ensemble_tta_accuracy:.4f}")

    # Visualiser quelques prédictions
    visualize_model_predictions(model, test_generator)

    # Analyser les confusions
    analyze_confusion_matrix(model, test_generator)


def safe_copy(src, dst):
    """Safely copy a file with error handling"""
    try:
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
        else:
            print(f"Warning: Source file does not exist: {src}")
    except Exception as e:
        print(f"Error copying {src} to {dst}: {e}")


# Create a feature fusion model
def create_fusion_model(num_classes, input_shape=(224, 224, 3)):
    # Image input branch
    img_input = Input(shape=input_shape, name='image_input')

    # Use EfficientNetB0 as the base model
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=img_input
    )

    # Freeze the base model initially
    base_model.trainable = False

    # CNN features
    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Handcrafted features input (25 features from extract_footprint_features)
    feature_input = Input(shape=(25,), name='feature_input')
    y = Dense(64, activation='relu')(feature_input)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y)

    # Combine CNN features and handcrafted features
    combined = tf.keras.layers.concatenate([x, y])
    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(0.4)(combined)

    # Output layer
    output = Dense(num_classes, activation='softmax')(combined)

    # Create model
    model = Model(inputs=[img_input, feature_input], outputs=output)

    return model, base_model


def weighted_ensemble_predictions(models, val_generator, test_generator):
    """Weight model predictions by their validation accuracy"""
    # Get validation accuracy for each model
    val_accuracies = []
    for model in models:
        _, val_acc = model.evaluate(val_generator)
        val_accuracies.append(val_acc)

    # Convert to weights (normalize)
    weights = np.array(val_accuracies) / sum(val_accuracies)
    print(f"Ensemble weights based on validation accuracy: {weights}")

    # Get predictions for test data
    all_preds = []
    for model in models:
        preds = model.predict(test_generator)
        all_preds.append(preds)

    # Weighted average
    weighted_preds = np.zeros_like(all_preds[0])
    for i, preds in enumerate(all_preds):
        weighted_preds += preds * weights[i]

    return weighted_preds, weights


def test_time_augmentation(model, image, num_augmentations=10):
    """Apply multiple augmentations at test time and average predictions"""
    aug_images = []

    # Original image
    aug_images.append(image)

    # Create augmented versions
    for _ in range(num_augmentations - 1):
        # Apply random augmentation
        aug_image = image.copy()
        if np.random.rand() > 0.5:
            aug_image = tf.image.flip_left_right(aug_image)
        if np.random.rand() > 0.5:
            aug_image = tf.image.flip_up_down(aug_image)

        # Random rotation (0, 90, 180, 270 degrees)
        k = np.random.randint(4)
        aug_image = tf.image.rot90(aug_image, k=k)

        aug_images.append(aug_image)

    # Stack all augmented images
    aug_images = np.stack(aug_images)

    # Get predictions for all augmentations
    predictions = model.predict(aug_images)

    # Average the predictions
    avg_prediction = np.mean(predictions, axis=0)

    return avg_prediction


def evaluate_with_tta(model, test_generator):
    """Évalue le modèle avec test-time augmentation"""
    all_preds = []
    all_labels = []

    for i in range(len(test_generator)):
        batch_x, batch_y = test_generator[i]
        batch_preds = []

        for j in range(len(batch_x)):
            img = batch_x[j]
            tta_pred = test_time_augmentation(model, img)
            batch_preds.append(tta_pred)

        all_preds.extend(batch_preds)
        all_labels.extend(batch_y)

    return np.array(all_preds), np.array(all_labels)


if __name__ == "__main__":
    tf.keras.mixed_precision.set_global_policy(
        'mixed_float16')  # Use mixed precision
    train_model()
