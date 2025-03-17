import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import os
from PIL import Image
import numpy as np


def load_data(data_dir):
    """
    Charge les images et leurs étiquettes (noms des espèces) à partir d'une structure de dossiers.
    Chaque sous-dossier est considéré comme une classe/espèce différente.
    """
    images = []
    labels = []

    print(f"Chargement des données depuis: {data_dir}")

    # Parcourir tous les sous-dossiers (espèces)
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            print(f"Traitement de l'espèce: {label}")
            count = 0

            # Parcourir tous les fichiers d'images dans le sous-dossier
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        images.append(preprocess_image(img_path))
                        labels.append(label)
                        count += 1
                    except Exception as e:
                        print(
                            f"Erreur lors du traitement de {img_path}: {str(e)}")

            print(f"  - {count} images chargées pour {label}")

    print(f"Total: {len(images)} images chargées, {len(set(labels))} espèces")
    return np.array(images), np.array(labels)


def preprocess_image(image_path):
    """
    Prétraite une image pour l'entraînement du modèle:
    - Redimensionnement à taille fixe (128x128)
    - Normalisation des valeurs de pixels (0-1)
    - Gestion des images en niveaux de gris
    """
    image = Image.open(image_path)
    image = image.resize((128, 128))  # Resize to a fixed size

    # Convertir en RGB si l'image est en niveaux de gris
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_array = np.array(image) / 255.0  # Normalize pixel values
    return image_array


def encode_labels(labels):
    """
    Encode les étiquettes textuelles (noms d'espèces) en valeurs numériques
    Retourne les étiquettes encodées et l'encodeur pour référence ultérieure
    """
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Afficher le mapping pour référence
    mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    print("Mapping des classes:")
    for idx, name in mapping.items():
        print(f"  {idx}: {name}")

    return encoded_labels, label_encoder


def split_data(images, labels, test_size=0.2, val_size=0.1, random_state=42):
    """
    Divise les données en ensembles d'entraînement, validation et test
    """
    # Première division: séparer les données de test
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Deuxième division: séparer les données d'entraînement et de validation
    # Recalculer val_size par rapport aux données restantes
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )

    print(
        f"Division des données: {len(X_train)} entraînement, {len(X_val)} validation, {len(X_test)} test")

    return X_train, X_val, X_test, y_train, y_val, y_test


def main(data_dir):
    """
    Fonction principale pour le prétraitement des données
    """
    # Charger les images et les étiquettes
    images, labels = load_data(data_dir)

    # Encoder les étiquettes
    encoded_labels, label_encoder = encode_labels(labels)

    # Diviser les données
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        images, encoded_labels)

    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder


if __name__ == "__main__":
    # Chemin vers le dossier contenant les sous-dossiers d'espèces
    data_directory = '../data/Mammifères'

    # Exécuter le prétraitement
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = main(
        data_directory)

    print(f"Prétraitement terminé. Données prêtes pour l'entraînement.")
    print(f"Forme des données d'entraînement: {X_train.shape}")
