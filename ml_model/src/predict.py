import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import argparse

# Configuration
IMG_SIZE = (299, 299)  # Même taille que celle utilisée lors de l'entraînement
# Chemin vers le modèle sauvegardé
MODEL_PATH = '../../models/footprint_classifier.keras'


def load_prediction_model():
    """
    Charge le modèle entraîné pour les prédictions
    """
    model_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), MODEL_PATH)
    if not os.path.exists(model_path):
        print(
            f"Erreur: Le modèle n'a pas été trouvé à l'emplacement: {model_path}")
        sys.exit(1)

    print(f"Chargement du modèle depuis {model_path}...")

    # Charger le modèle en ignorant les objets personnalisés
    model = tf.keras.models.load_model(
        model_path,
        compile=False  # Ne pas charger l'optimiseur et la fonction de perte
    )
    print("Modèle chargé avec succès!")

    # Charger les indices des classes si disponibles
    class_indices_path = os.path.join(
        os.path.dirname(model_path), 'class_indices.npy')
    if os.path.exists(class_indices_path):
        class_indices = np.load(class_indices_path, allow_pickle=True).item()
    else:
        # Si le fichier n'existe pas, essayons de trouver les noms de classes à partir des logs
        print("Attention: Fichier d'indices de classes non trouvé.")
        class_indices = {
            0: 'Castor', 1: 'Cerf', 2: 'Cheval', 3: 'Coyote', 4: 'Dindon',
            5: 'Écureuil', 6: 'Éléphant', 7: 'Lion', 8: 'Loutre', 9: 'Lynx',
            10: 'Moufette', 11: 'Oie', 12: 'Ours', 13: 'Rat', 14: 'Raton laveur',
            15: 'Renard', 16: 'Souris', 17: 'Vison'
        }

    return model, class_indices


def preprocess_image(image_path):
    """
    Prétraite l'image pour la prédiction
    """
    if not os.path.exists(image_path):
        print(
            f"Erreur: L'image n'a pas été trouvée à l'emplacement: {image_path}")
        sys.exit(1)

    print(f"Prétraitement de l'image: {image_path}")

    try:
        # Charger et redimensionner l'image
        img = load_img(image_path, target_size=IMG_SIZE)
        # Convertir en tableau numpy et normaliser
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalisation
        # Ajouter une dimension pour le batch
        img_array = np.expand_dims(img_array, axis=0)

        return img_array, img

    except Exception as e:
        print(f"Erreur lors du prétraitement de l'image: {str(e)}")
        sys.exit(1)


def predict_image(model, img_array, class_indices, top_k=5):
    """
    Fait une prédiction sur l'image et renvoie les top_k classes
    """
    print("Prédiction en cours...")
    try:
        # Faire la prédiction
        predictions = model.predict(img_array)

        # Obtenir les indices des top_k classes
        top_indices = predictions[0].argsort()[-top_k:][::-1]

        # Créer la liste des résultats
        results = []
        for i in top_indices:
            class_name = class_indices.get(i, f"Classe_{i}")
            confidence = predictions[0][i]
            results.append((class_name, confidence))

        return results

    except Exception as e:
        print(f"Erreur lors de la prédiction: {str(e)}")
        sys.exit(1)


def display_results(img, results):
    """
    Affiche l'image et les résultats de la prédiction
    """
    # Afficher les résultats dans la console
    print("\nRésultats de la prédiction:")
    print("--------------------------")
    for i, (class_name, confidence) in enumerate(results):
        print(f"{i+1}. {class_name}: {confidence*100:.2f}%")

    # Afficher l'image et les résultats dans une fenêtre
    plt.figure(figsize=(10, 6))

    # Afficher l'image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Image d'empreinte")
    plt.axis('off')

    # Afficher les résultats sous forme de graphique
    plt.subplot(1, 2, 2)
    class_names = [class_name for class_name, _ in results]
    confidences = [confidence * 100 for _, confidence in results]

    bars = plt.barh(range(len(class_names)), confidences, color='skyblue')
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel('Confiance (%)')
    plt.title('Top 5 prédictions')
    plt.xlim(0, 100)

    # Ajouter les pourcentages aux barres
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f'{confidences[i]:.1f}%', va='center')

    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.show()


def main():
    """
    Fonction principale
    """
    # Parser les arguments de ligne de commande
    parser = argparse.ArgumentParser(
        description="Prédiction d'empreintes animales")
    parser.add_argument(
        "image_path", help="Chemin vers l'image d'empreinte à analyser")
    args = parser.parse_args()

    # Charger le modèle
    model, class_indices = load_prediction_model()

    # Prétraiter l'image
    img_array, img = preprocess_image(args.image_path)

    # Faire la prédiction
    results = predict_image(model, img_array, class_indices)

    # Afficher les résultats
    display_results(img, results)


if __name__ == "__main__":
    main()
