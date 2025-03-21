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
from tensorflow.keras.applications import EfficientNetB0
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Réduire la verbosité des logs


# Ajouter en haut du fichier

# Configuration
# Résolution plus élevée pour une meilleure discrimination
IMG_SIZE = (299, 299)
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001
BASE_MODEL = 'resnet'  # 'efficientnet'  # 'mobilenet'
RANDOM_STATE = 42
MODEL_SAVE_PATH = '../models/footprint_classifier.keras'

# Créer le dossier pour sauvegarder le modèle si nécessaire
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Ajouter en haut du fichier la fonction de perte focale


# Modifier la fonction focal_loss pour favoriser davantage le rappel
def focal_loss(gamma=2., alpha=.4):
    """
    Focal Loss ajustée pour favoriser davantage le rappel
    avec gestion explicite des types
    """
    def focal_loss_fixed(y_true, y_pred):
        # Conversion explicite des types
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Clip pour éviter les NaNs
        epsilon = tf.constant(1e-7, dtype=tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Calcul de focal loss avec plus d'accent sur les faux négatifs
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Utiliser tf.pow au lieu de ** pour assurer compatibilité types
        modulating_factor = tf.pow(1.0 - y_pred, gamma)

        # Utiliser des multiplicateurs scalaires simples
        weighted_cross_entropy = alpha * modulating_factor * cross_entropy

        # Somme réduite par axe
        loss = tf.reduce_sum(weighted_cross_entropy, axis=-1)
        return loss

    return focal_loss_fixed


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
    Crée des générateurs de données avec augmentation
    """
    # Définir les classes qui ont un faible rappel
    low_recall_classes = ['Raton laveur', 'Renard', 'Lynx', 'Écureuil']

    # Utiliser les fonctions d'augmentation intégrées
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=0.35,
        height_shift_range=0.35,
        shear_range=0.35,
        zoom_range=0.35,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.65, 1.35],
        fill_mode='reflect'
    )

    # Datagen simple pour validation et test
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # Générateurs standards
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='',
        x_col='image_path',
        y_col='species',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    # Création du générateur de validation
    val_generator = val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory='',
        x_col='image_path',
        y_col='species',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # Ne pas mélanger pour la validation
    )

    # Création du générateur de test
    test_generator = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory='',
        x_col='image_path',
        y_col='species',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # Ne pas mélanger pour le test
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
    elif base_model_type == 'efficientnet':
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*IMG_SIZE, 3)
        )
    else:
        raise ValueError(f"Type de modèle de base inconnu: {base_model_type}")

    # Geler les couches du modèle de base SAUF les derniers blocs
    # Pour MobileNetV2, dégeler les derniers blocs convolutional
    # Dégeler les dernières couches (environ 20% du modèle)
    fine_tune_at = len(base_model.layers) * 0.8

    for layer in base_model.layers[:int(fine_tune_at)]:
        layer.trainable = False

    # Construire le modèle complet avec des couches plus larges
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),  # Augmenter la taille à 1024
        # Augmenter le dropout à 0.5 pour éviter le surapprentissage
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.3),
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
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            min_lr=1e-7
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
        metrics=['accuracy', tf.keras.metrics.Precision(
        ), tf.keras.metrics.Recall()]  # Ajouter des métriques
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


def evaluate_model(model, test_generator, class_names, adjust=True):
    """
    Évalue le modèle sur l'ensemble de test
    """
    print("\nÉvaluation du modèle sur l'ensemble de test...")
    # Récupérer toutes les métriques retournées par model.evaluate
    evaluation_results = model.evaluate(test_generator)

    # Extraire les métriques individuelles
    if isinstance(evaluation_results, list):
        # Si plusieurs métriques sont retournées
        test_loss = evaluation_results[0]
        test_accuracy = evaluation_results[1]
        # Vous pouvez également extraire precision et recall si nécessaire
        if len(evaluation_results) > 2:
            test_precision = evaluation_results[2]
            test_recall = evaluation_results[3] if len(
                evaluation_results) > 3 else None
            print(f"Précision (precision): {test_precision:.4f}")
            if test_recall:
                print(f"Rappel (recall): {test_recall:.4f}")
    else:
        # Si une seule métrique est retournée
        test_loss = evaluation_results
        test_accuracy = None

    print(f"Perte sur l'ensemble de test: {test_loss:.4f}")
    if test_accuracy:
        print(f"Précision sur l'ensemble de test: {test_accuracy:.4f}")

    # Prédictions sur l'ensemble de test
    y_pred_probs = model.predict(test_generator)

    # Ajuster les prédictions si demandé
    if adjust:
        difficult_classes = ['Raton laveur', 'Renard', 'Lynx', 'Écureuil']
        y_pred_probs = adjust_predictions(
            y_pred_probs, difficult_classes, test_generator.class_indices)

    y_pred = np.argmax(y_pred_probs, axis=1)

    # Véritable classe
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
    plt.plot(history['accuracy'], label='Entraînement')
    plt.plot(history['val_accuracy'], label='Validation')
    plt.title('Précision du modèle')
    plt.xlabel('Epoch')
    plt.ylabel('Précision')
    plt.legend()

    # Graphique de perte
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Entraînement')
    plt.plot(history['val_loss'], label='Validation')
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


# Augmenter le rappel pour les classes problématiques
def adjust_predictions(predictions, difficult_classes, class_indices):
    # Indices des classes difficiles
    diff_indices = [class_indices[cls_name] for cls_name in difficult_classes]

    # Facteur de boost pour ces classes
    boost_factor = 1.2

    # Appliquer le boost
    adjusted_preds = predictions.copy()
    adjusted_preds[:, diff_indices] *= boost_factor

    return adjusted_preds


def visualize_augmentations(train_df, difficult_classes=['Raton laveur', 'Renard']):
    """
    Visualise les augmentations appliquées aux images
    """
    # Sélectionner quelques images des classes difficiles
    sample_images = []
    for cls in difficult_classes:
        samples = train_df[train_df['species'] == cls].sample(
            min(2, len(train_df[train_df['species'] == cls])))
        sample_images.extend(samples['image_path'].tolist())

    # Si aucune image des classes difficiles n'est trouvée, prendre des images aléatoires
    if not sample_images:
        sample_images = train_df.sample(4)['image_path'].tolist()

    # Visualiser les augmentations
    plt.figure(figsize=(15, 5 * len(sample_images)))

    for i, img_path in enumerate(sample_images):
        # Charger l'image
        img = load_img(img_path, target_size=IMG_SIZE)
        img = img_to_array(img)

        # Créer une grille de 3x3 pour chaque image
        plt.subplot(len(sample_images), 4, i*4 + 1)
        plt.imshow(img.astype('uint8'))
        plt.title(f"Original: {os.path.basename(img_path)}")
        plt.axis('off')

        # Appliquer différentes augmentations
        for j in range(3):
            aug_img = augment_image(img.copy(), difficult_classes, None)
            plt.subplot(len(sample_images), 4, i*4 + j + 2)
            plt.imshow(aug_img.numpy().astype('uint8'))
            plt.title(f"Augmentation {j+1}")
            plt.axis('off')

    plt.tight_layout()
    plt.savefig('augmentation_examples.png')
    plt.show()


def main():
    """
    Fonction principale avec une stratégie d'entraînement en deux phases
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

    # Phase 1: Entraîner uniquement les couches supérieures
    print("\n--- Phase 1: Entraînement des couches supérieures ---")

    # Geler complètement le modèle de base
    for layer in model.layers[0].layers:
        layer.trainable = False

    # Compiler avec un taux d'apprentissage initial plus élevé
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Bon choix actuel
        loss=focal_loss(gamma=2.0),  # Remplacer categorical_crossentropy
        metrics=['accuracy', tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall()]
    )

    # Entraîner pendant quelques époques
    print("\nPhase 1: Entraînement des couches supérieures...")
    history1 = model.fit(
        train_generator,
        epochs=15,
        validation_data=val_generator,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3),
            ModelCheckpoint(
                filepath=MODEL_SAVE_PATH.replace('.keras', '_phase1.keras'),
                monitor='val_accuracy',
                save_best_only=True
            )
        ],
        class_weight=class_weights,
        verbose=1
    )

    # Phase 2: Fine-tuning du modèle complet
    print("\n--- Phase 2: Fine-tuning du modèle complet ---")

    # Dégeler d'abord le dernier bloc résiduel
    for layer in model.layers[0].layers:
        if 'conv5' in layer.name:  # Dernier bloc résiduel dans ResNet
            layer.trainable = True
        else:
            layer.trainable = False

    # Puis après quelques époques, dégeler plus de blocs
    # Pour un dégel progressif plus fin après la phase 1

    # Recompiler avec un taux d'apprentissage plus faible
    model.compile(
        # Considérez une valeur plus faible (5e-5)
        optimizer=Adam(learning_rate=0.00005),
        loss=focal_loss(gamma=2.0),  # Remplacer categorical_crossentropy
        metrics=['accuracy', tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall()]
    )

    # Entraîner pendant plus d'époques
    print("\nPhase 2: Fine-tuning du modèle complet...")
    history2 = model.fit(
        train_generator,
        epochs=35,
        validation_data=val_generator,
        callbacks=[
            ModelCheckpoint(filepath=MODEL_SAVE_PATH,
                            monitor='val_accuracy', save_best_only=True),
            EarlyStopping(monitor='val_loss', patience=10,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=5, min_lr=1e-7)
        ],
        class_weight=class_weights,
        verbose=1
    )

    # Combiner les historiques pour les visualisations
    history = {}
    for k in history1.history:
        if k in history2.history:
            history[k] = history1.history[k] + history2.history[k]

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

    # Visualiser les augmentations
    visualize_augmentations(
        train_df, ['Raton laveur', 'Renard', 'Lynx', 'Écureuil'])

    print("\nEntraînement terminé avec succès!")


if __name__ == "__main__":
    # Définir les configurations de seed pour la reproductibilité
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    # Avant d'exécuter le main
    tf.keras.backend.clear_session()

    # Exécuter le pipeline principal
    main()
