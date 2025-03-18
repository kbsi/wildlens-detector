import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0005
BASE_MODEL = 'mobilenet'
RANDOM_STATE = 42
MODEL_SAVE_PATH = '../models/footprint_classifier.keras'

# Créer le dossier pour sauvegarder le modèle si nécessaire
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)


def load_dataset(csv_path):
    """
    Charge le dataset à partir du fichier CSV
    """
    print("Chargement du dataset...")
    df = pd.read_csv(csv_path)
    print(f"Dataset chargé avec succès: {len(df)} images")

    # Afficher la distribution des classes
    class_distribution = df['species'].value_counts()
    print("Distribution des classes:")
    for species, count in class_distribution.items():
        print(f"- {species}: {count} images")

    return df


def preprocess_data(df, base_dir='../data/Mammifères_augmented/'):
    """
    Prétraite les données et divise en ensembles train/val/test
    """
    # Assurer que le chemin de base est correct - remonter d'un niveau depuis le répertoire src
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.normpath(os.path.join(script_dir, base_dir))
    print(f"Recherche des images dans: {base_dir}")

    # Filtrer les lignes avec des chemins d'images existants
    valid_images = []
    modified_paths = []  # Liste pour stocker les chemins modifiés

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Vérification des images"):
        img_path = os.path.join(base_dir, row['image_path'])
        if os.path.exists(img_path):
            valid_images.append(idx)
            # Stocker le chemin complet plutôt que le chemin relatif
            modified_paths.append(img_path)
        else:
            # Afficher quelques exemples de chemins non trouvés pour déboguer
            if len(valid_images) == 0 and idx < 5:
                print(f"Image non trouvée: {img_path}")

    # Créer un nouveau DataFrame avec les chemins complets
    df_filtered = df.loc[valid_images].reset_index(drop=True)
    # Remplacer les chemins relatifs par les chemins complets
    df_filtered['image_path'] = modified_paths
    print(f"Images valides: {len(df_filtered)}/{len(df)}")

    # Si aucune image valide n'est trouvée, afficher un message d'erreur explicite
    if len(df_filtered) == 0:
        raise ValueError(
            "Aucune image valide trouvée ! Vérifiez le chemin de base et la structure des dossiers.")

    # Encodage des étiquettes
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df_filtered['species'])
    class_names = label_encoder.classes_
    print(f"Classes: {class_names}")

    # Division en ensembles d'entraînement et de test
    train_df, test_df = train_test_split(
        df_filtered,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df_filtered['species']
    )

    # Division de l'ensemble d'entraînement en entraînement et validation
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.25,  # 0.25 * 0.8 = 0.2 de l'ensemble total
        random_state=RANDOM_STATE,
        stratify=train_df['species']
    )

    print(f"Ensemble d'entraînement: {len(train_df)} images")
    print(f"Ensemble de validation: {len(val_df)} images")
    print(f"Ensemble de test: {len(test_df)} images")

    return train_df, val_df, test_df, class_names, label_encoder


def create_data_generators(train_df, val_df, test_df, base_dir=''):
    """
    Crée des générateurs de données pour l'entraînement, la validation et le test
    """
    # Augmentation des données pour l'entraînement
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

    # Normalisation uniquement pour la validation et le test
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # Assurer que le chemin de base est correct - remonter d'un niveau depuis le répertoire src
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.normpath(os.path.join(script_dir, base_dir))

    # Générateurs de données - utiliser une chaîne vide comme directory car les chemins sont complets
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='',  # Utiliser une chaîne vide car les chemins sont déjà complets
        x_col='image_path',
        y_col='species',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory='',  # Utiliser une chaîne vide car les chemins sont déjà complets
        x_col='image_path',
        y_col='species',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory='',  # Utiliser une chaîne vide car les chemins sont déjà complets
        x_col='image_path',
        y_col='species',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator


def build_model(num_classes, base_model_type=BASE_MODEL):
    """
    Construit le modèle CNN pour la classification
    """
    print(f"Construction du modèle avec {base_model_type} comme base...")

    if base_model_type == 'mobilenet':
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*IMG_SIZE, 3)
        )
    elif base_model_type == 'resnet':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*IMG_SIZE, 3)
        )
    else:
        raise ValueError(f"Type de modèle de base inconnu: {base_model_type}")

    # Geler les couches du modèle de base
    base_model.trainable = False

    # Construire le modèle complet
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    return model


def calculate_class_weights(train_df, train_generator):
    """
    Calcule les poids des classes en fonction de leur fréquence
    Les classes moins représentées reçoivent un poids plus élevé
    """
    # Compter le nombre d'échantillons par classe
    class_counts = train_df['species'].value_counts()
    total_samples = len(train_df)
    n_classes = len(class_counts)

    # Calculer les poids: inversement proportionnels à la fréquence de la classe
    class_weights = {}
    for class_name, count in class_counts.items():
        # Formule: total_samples / (n_classes * count)
        # Cette formule donne plus de poids aux classes minoritaires
        weight = total_samples / (n_classes * count)
        # Conversion en indice numérique pour Keras
        class_idx = train_generator.class_indices.get(class_name)
        if class_idx is not None:
            class_weights[class_idx] = weight

    print("\nPoids des classes:")
    for class_idx, weight in sorted(class_weights.items()):
        class_name = next((name for name, idx in train_generator.class_indices.items(
        ) if idx == class_idx), f"Classe {class_idx}")
        print(f"- {class_name}: {weight:.4f}")

    return class_weights


def train_model(model, train_generator, val_generator, class_names, class_weights=None):
    """
    Entraîne le modèle et affiche les résultats
    """
    # Définir les callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        ),
        ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Compiler le modèle
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Résumé du modèle
    model.summary()

    print("\nEntraînement du modèle...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weights,  # Utilisation des poids des classes
        verbose=1
    )

    return model, history


def evaluate_model(model, test_generator, class_names):
    """
    Évalue le modèle sur l'ensemble de test
    """
    print("\nÉvaluation du modèle sur l'ensemble de test...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Perte sur l'ensemble de test: {test_loss:.4f}")
    print(f"Précision sur l'ensemble de test: {test_accuracy:.4f}")

    # Prédictions sur l'ensemble de test
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Véritable classe (peut nécessiter une adaptation selon la façon dont le générateur renvoie les étiquettes)
    y_true = test_generator.classes

    # Rapport de classification
    print("\nRapport de classification:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    return y_true, y_pred


def plot_training_curves(history):
    """
    Affiche les courbes d'entraînement (précision et perte)
    """
    # Créer une figure avec deux sous-graphiques
    plt.figure(figsize=(12, 5))

    # Graphique de précision
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entraînement')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Précision du modèle')
    plt.xlabel('Epoch')
    plt.ylabel('Précision')
    plt.legend()

    # Graphique de perte
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entraînement')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Perte du modèle')
    plt.xlabel('Epoch')
    plt.ylabel('Perte')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Affiche la matrice de confusion
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Matrice de confusion')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()


def main():
    """
    Fonction principale exécutant le pipeline d'entraînement complet
    """
    # Chemin vers le fichier CSV des données
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.normpath(os.path.join(
        script_dir, '../data/footprint_dataset_augmented.csv'))
    print(f"Utilisation du fichier CSV: {csv_path}")

    # 1. Chargement du dataset
    df = load_dataset(csv_path)

    # 2. Prétraitement des données
    train_df, val_df, test_df, class_names, label_encoder = preprocess_data(df)

    # 3. Création des générateurs de données
    train_generator, val_generator, test_generator = create_data_generators(
        train_df, val_df, test_df)

    # 4. Calcul des poids des classes pour gérer le déséquilibre
    class_weights = calculate_class_weights(train_df, train_generator)

    # 5. Construction du modèle
    model = build_model(len(class_names))

    # 6. Entraînement du modèle avec les poids des classes
    model, history = train_model(
        model, train_generator, val_generator, class_names, class_weights)

    # 7. Évaluation du modèle
    y_true, y_pred = evaluate_model(model, test_generator, class_names)

    # 8. Visualisation des résultats
    plot_training_curves(history)
    plot_confusion_matrix(y_true, y_pred, class_names)

    # 9. Sauvegarde du mapping des classes
    class_indices = {v: k for k, v in train_generator.class_indices.items()}
    # Correction du chemin pour la sauvegarde des indices de classes
    np.save(os.path.join(os.path.dirname(MODEL_SAVE_PATH),
            'class_indices.npy'), class_indices)
    print(f"Modèle et indices de classes sauvegardés dans {MODEL_SAVE_PATH}")

    print("\nEntraînement terminé avec succès!")


if __name__ == "__main__":
    # Définir les configurations de seed pour la reproductibilité
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    # Exécuter le pipeline principal
    main()
