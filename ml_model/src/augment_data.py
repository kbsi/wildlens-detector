#!/usr/bin/env python3
"""
Script pour enrichir le jeu de données d'empreintes animales par augmentation de données
en utilisant TensorFlow, compatible avec les versions récentes de NumPy.
"""

import pandas as pd
import os
import numpy as np
import cv2
import glob
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
BASE_PATH = 'data/Mammifères/'
CSV_PATH = 'data/footprint_dataset.csv'
OUTPUT_PATH = 'data/Mammifères_augmented/'
# Nombre d'images augmentées à générer par image originale
AUGMENTATIONS_PER_IMAGE = 5

# Créer les dossiers de sortie s'ils n'existent pas
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Charger les données existantes
if os.path.exists(CSV_PATH):
    existing_data = pd.read_csv(CSV_PATH)
else:
    print(
        f"Le fichier CSV {CSV_PATH} n'existe pas. Exécutez d'abord generate-csv.py.")
    exit(1)

# Définition des augmentations avec TensorFlow


def augment_image(image):
    """Applique une série d'augmentations à l'image fournie."""
    # Convertir en tensor et normaliser
    image = tf.cast(image, tf.float32) / 255.0

    # Transformations spatiales (rotation, flips, etc.)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)

    # Rotation aléatoire (TensorFlow n'a pas de rotation arbitraire simple,
    # mais nous pouvons utiliser rot90 pour des angles de 90°)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)

    # Zoom et recadrage aléatoires
    # D'abord nous faisons un crop central avec un facteur aléatoire
    crop_factor = tf.random.uniform(shape=[], minval=0.8, maxval=1.0)
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    image = tf.image.central_crop(image, crop_factor)
    # Puis nous redimensionnons à la taille originale
    image = tf.image.resize(image, [h, w])

    # Transformation de perspective (déplacement)
    # TensorFlow n'a pas directement de transformation de perspective,
    # mais nous pouvons simuler des déplacements en coupant et en redimensionnant
    if tf.random.uniform(()) > 0.7:
        # Décalage aléatoire horizontal ou vertical
        offset_h = tf.random.uniform(
            shape=[], minval=-0.1, maxval=0.1) * tf.cast(h, tf.float32)
        offset_w = tf.random.uniform(
            shape=[], minval=-0.1, maxval=0.1) * tf.cast(w, tf.float32)
        offset_h, offset_w = tf.cast(
            offset_h, tf.int32), tf.cast(offset_w, tf.int32)

        # Appliquer le décalage et redimensionner
        if offset_h > 0:
            image = tf.image.crop_to_bounding_box(
                image, offset_h, 0, h-offset_h, w)
            image = tf.image.resize(image, [h, w])
        elif offset_h < 0:
            image = tf.image.crop_to_bounding_box(image, 0, 0, h+offset_h, w)
            image = tf.image.resize(image, [h, w])

        if offset_w > 0:
            image = tf.image.crop_to_bounding_box(
                image, 0, offset_w, h, w-offset_w)
            image = tf.image.resize(image, [h, w])
        elif offset_w < 0:
            image = tf.image.crop_to_bounding_box(image, 0, 0, h, w+offset_w)
            image = tf.image.resize(image, [h, w])

    # Modifications de couleur/luminosité/contraste
    # Luminosité
    image = tf.image.random_brightness(image, max_delta=0.2)
    # Contraste
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # Saturation (uniquement pour les images RGB)
    if tf.shape(image)[-1] == 3:
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)

    # Ajout de bruit (simulation de texture de sol)
    if tf.random.uniform(()) > 0.5:
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
        image = tf.clip_by_value(image + noise, 0.0, 1.0)

    # Convertir en uint8 pour l'enregistrement
    image = tf.cast(image * 255.0, tf.uint8)
    return image

# Fonction pour visualiser une augmentation


def visualize_augmentation(image_path, n_samples=5):
    """Visualise les résultats d'augmentation pour une image donnée."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    augmented_images = [img]  # Commencer avec l'image originale

    # Générer des exemples augmentés
    for _ in range(n_samples):
        aug_img = augment_image(img).numpy()
        augmented_images.append(aug_img)

    # Afficher les images
    fig, axes = plt.subplots(1, n_samples+1, figsize=(15, 4))
    axes[0].imshow(augmented_images[0])
    axes[0].set_title("Original")
    axes[0].axis('off')

    for i in range(1, n_samples+1):
        axes[i].imshow(augmented_images[i])
        axes[i].set_title(f"Augmented {i}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# Liste pour stocker les nouvelles données
new_data_rows = []

# Parcourir chaque espèce
species_folders = [d for d in os.listdir(
    BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]
print(f"Traitement de {len(species_folders)} espèces...")

for species in species_folders:
    species_path = os.path.join(BASE_PATH, species)
    output_species_path = os.path.join(OUTPUT_PATH, species)

    # Créer le dossier de sortie pour cette espèce s'il n'existe pas
    if not os.path.exists(output_species_path):
        os.makedirs(output_species_path)

    # Trouver toutes les images originales
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(os.path.join(species_path, ext)))

    print(f"  {species}: {len(image_paths)} images originales trouvées")

    # Pour chaque image, générer des versions augmentées
    for img_path in tqdm(image_paths, desc=f"Augmentation {species}", unit="img"):
        # Lire l'image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Impossible de lire l'image: {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Obtenir le nom de fichier de base sans extension
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # Copier l'original dans le dossier augmenté
        original_dest_path = os.path.join(
            output_species_path, f"{base_name}_original.png")
        cv2.imwrite(original_dest_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Ajouter l'original à notre nouvelle liste de données
        rel_path = os.path.relpath(
            original_dest_path, start=os.path.dirname(OUTPUT_PATH))
        new_data_rows.append({
            'image_path': rel_path,
            'species': species,
            'augmented': False
        })

        # Générer des versions augmentées
        for i in range(AUGMENTATIONS_PER_IMAGE):
            # Appliquer l'augmentation
            aug_img = augment_image(img).numpy()

            # Sauvegarder l'image augmentée
            aug_path = os.path.join(
                output_species_path, f"{base_name}_aug{i+1}.png")
            cv2.imwrite(aug_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

            # Ajouter à notre liste de données
            rel_path = os.path.relpath(
                aug_path, start=os.path.dirname(OUTPUT_PATH))
            new_data_rows.append({
                'image_path': rel_path,
                'species': species,
                'augmented': True
            })

# Créer le nouveau DataFrame
augmented_data = pd.DataFrame(new_data_rows)

# Enregistrer en CSV
csv_augmented_path = 'data/footprint_dataset_augmented.csv'
augmented_data.to_csv(csv_augmented_path, index=False)

# Statistiques
original_count = len(augmented_data[augmented_data['augmented'] == False])
augmented_count = len(augmented_data[augmented_data['augmented'] == True])
total_count = len(augmented_data)

print(f"\nAugmentation terminée:")
print(f"  Images originales: {original_count}")
print(f"  Images augmentées: {augmented_count}")
print(f"  Total: {total_count}")
print(f"  Expansion du jeu de données: {total_count / original_count:.1f}x")
print(f"Dataset enregistré sous: {os.path.abspath(csv_augmented_path)}")

# Visualiser quelques exemples
if image_paths:
    print("\nVisualisation de quelques exemples d'augmentation:")
    for _ in range(3):  # Montrer 3 exemples
        random_image = np.random.choice(image_paths)
        print(f"Exemple d'augmentation pour: {os.path.basename(random_image)}")
        visualize_augmentation(random_image)
