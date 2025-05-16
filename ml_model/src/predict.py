import os
import sys
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse
import cv2
import glob

# Ajouter l'import de la classe du modèle
from model import FootprintClassifier  # Remplacez par le bon import

# Configuration
IMG_SIZE = (224, 224)  # Taille utilisée lors de l'entraînement
# Chemin vers le modèle sauvegardé
MODEL_PATH = '../../models/footprint_classifier.pth'


def load_prediction_model():
    """
    Charge le modèle entraîné pour les prédictions (PyTorch)
    """
    model_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), MODEL_PATH)
    if not os.path.exists(model_path):
        print(
            f"Erreur: Le modèle n'a pas été trouvé à l'emplacement: {model_path}")
        sys.exit(1)

    print(f"Chargement du modèle depuis {model_path}...")

    # Charger le modèle PyTorch
    model = FootprintClassifier()  # Remplacez par le bon nom de classe et les arguments
    
    # Load the entire checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Check if the checkpoint contains a 'model_state_dict' key
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    
    model.eval()  # Mettre le modèle en mode évaluation
    
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
            0: 'castor', 1: 'cerf_mulet', 2: 'cheval', 3: 'coyote', 4: 'dinde',
            5: 'ecureuil_gris_occidental', 6: 'elephant', 7: 'lion', 8: 'loutre', 9: 'lynx_roux',
            10: 'moufette', 11: 'oie', 12: 'ours_noir', 13: 'rat', 14: 'raton_laveur',
            15: 'renard_gris', 16: 'souris', 17: 'vison'
        }

    return model, class_indices


def preprocess_image(image_path):
    """
    Prétraite l'image pour la prédiction (PyTorch)
    """
    if not os.path.exists(image_path):
        print(
            f"Erreur: L'image n'a pas été trouvée à l'emplacement: {image_path}")
        sys.exit(1)

    print(f"Prétraitement de l'image: {image_path}")

    try:
        # Charger l'image avec PIL
        img = Image.open(image_path).convert('RGB')

        # Définir les transformations
        preprocess = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])  # Normalisation ImageNet
        ])

        # Appliquer les transformations
        img_tensor = preprocess(img)

        # Ajouter une dimension pour le batch
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor, img

    except Exception as e:
        print(f"Erreur lors du prétraitement de l'image: {str(e)}")
        sys.exit(1)


def predict_image(model, img_tensor, class_indices, top_k=5):
    """
    Fait une prédiction sur l'image et renvoie les top_k classes (PyTorch)
    """
    print("Prédiction en cours...")
    try:
        # Désactiver le calcul du gradient
        with torch.no_grad():
            # Faire la prédiction
            outputs = model(img_tensor)
            # Appliquer softmax pour obtenir les probabilités
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # Obtenir les indices des top_k classes
        top_indices = torch.topk(probabilities, top_k).indices.cpu().numpy()

        # Créer la liste des résultats
        results = []
        for i in top_indices:
            class_name = class_indices.get(i, f"Classe_{i}")
            confidence = probabilities[i].item()
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


def preprocess_footprint_image(img_path):
    """
    Prétraite l'image d'empreinte pour la prédiction.
    Cette fonction doit correspondre au prétraitement utilisé pendant l'entraînement.
    
    Args:
        img_path (str): Chemin vers l'image à prétraiter
    
    Returns:
        array: Image prétraitée de taille (224, 224, 3)
    """
    # Charger l'image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Erreur: Impossible de charger l'image {img_path}")
        return None

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Amélioration du contraste via CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. Débruitage bilatéral
    denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    # 3. Détection des contours avec Canny
    edges = cv2.Canny(denoised, 50, 150)

    # 4. Fermeture morphologique pour connecter les contours proches
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. Trouver et remplir les contours principaux
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

    # Convertir en RGB (pour correspondre à l'entrée du modèle)
    result_rgb = np.stack([result, result, result], axis=-1)

    return result_rgb


def load_species_labels(labels_path="data/species_labels.txt"):
    """
    Charge les noms des espèces depuis un fichier.
    Si le fichier n'existe pas, utilise une liste par défaut.
    
    Args:
        labels_path (str): Chemin vers le fichier contenant les noms des espèces
        
    Returns:
        list: Liste des noms d'espèces
    """
    try:
        with open(labels_path, 'r') as f:
            species_labels = [line.strip() for line in f.readlines()]
        return species_labels
    except FileNotFoundError:
        # Liste d'exemple, à remplacer par vos propres classes
        print(f"Fichier d'étiquettes non trouvé à {labels_path}, utilisation des étiquettes par défaut.")
        return ["cerf", "chevreuil", "sanglier", "renard", "blaireau", "lapin", "lievre", "ecureuil"]


def predict_single_image(model, img_path, species_labels):
    """
    Prédit l'espèce pour une seule image.
    
    Args:
        model: Modèle PyTorch chargé
        img_path (str): Chemin vers l'image à prédire
        species_labels (list): Liste des noms d'espèces
    
    Returns:
        tuple: (classe prédite (str), prédictions (array), image prétraitée (array))
    """
    # Prétraiter l'image
    preprocessed_img = preprocess_footprint_image(img_path)
    if preprocessed_img is None:
        return None, None, None
    
    # Adapter l'image pour le modèle
    img_array = np.expand_dims(preprocessed_img, axis=0)
    
    # Faire la prédiction
    predictions = model.predict(img_array)
    
    # Obtenir l'indice de la classe avec la plus haute probabilité
    predicted_class_idx = np.argmax(predictions[0])
    
    # Correspondance avec le nom de l'espèce
    if predicted_class_idx < len(species_labels):
        predicted_species = species_labels[predicted_class_idx]
    else:
        predicted_species = f"Classe {predicted_class_idx} (inconnue)"
    
    return predicted_species, predictions[0], preprocessed_img


def visualize_prediction(img_path, predicted_species, predictions, preprocessed_img, species_labels):
    """
    Visualise l'image originale, l'image prétraitée et les prédictions.
    
    Args:
        img_path (str): Chemin vers l'image originale
        predicted_species (str): Espèce prédite
        predictions (array): Probabilités prédites pour chaque classe
        preprocessed_img (array): Image prétraitée
        species_labels (list): Liste des noms d'espèces
    """
    # Charger l'image originale
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    # Créer une figure avec 3 sous-graphiques
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Afficher l'image originale
    axs[0].imshow(orig_img)
    axs[0].set_title("Image originale")
    axs[0].axis('off')
    
    # Afficher l'image prétraitée
    axs[1].imshow(preprocessed_img)
    axs[1].set_title("Image prétraitée")
    axs[1].axis('off')
    
    # Afficher le graphique des prédictions
    # Limiter aux 5 meilleures prédictions pour la lisibilité
    top_indices = np.argsort(predictions)[-5:][::-1]
    top_species = [species_labels[i] if i < len(species_labels) else f"Classe {i}" for i in top_indices]
    top_probas = [predictions[i] for i in top_indices]
    
    barplot = axs[2].barh(top_species, top_probas, color='skyblue')
    axs[2].set_xlabel('Probabilité')
    axs[2].set_title('Top 5 prédictions')
    
    # Ajouter les valeurs sur les barres
    for i, v in enumerate(top_probas):
        axs[2].text(v + 0.01, i, f'{v:.2f}', color='black', va='center')
    
    # Ajuster l'échelle des x
    axs[2].set_xlim(0, 1.1)
    
    # Colorer la barre de la meilleure prédiction
    barplot[0].set_color('green')
    
    # Titre global
    plt.suptitle(f"Prédiction: {predicted_species}", fontsize=16)
    
    plt.tight_layout()
    plt.show()


def test_model_on_folder(model_path, image_folder, labels_path=None):
    """
    Teste le modèle sur toutes les images d'un dossier.
    
    Args:
        model_path (str): Chemin vers le modèle entraîné
        image_folder (str): Dossier contenant les images à tester
        labels_path (str): Chemin vers le fichier contenant les noms des espèces
    """
    # Charger le modèle
    print(f"Chargement du modèle depuis {model_path}...")
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    
    # Charger les étiquettes d'espèces
    species_labels = load_species_labels(labels_path)
    print(f"Classes chargées: {species_labels}")
    
    # Chercher les images
    supported_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for format in supported_formats:
        image_paths.extend(glob.glob(os.path.join(image_folder, format)))
    
    if not image_paths:
        print(f"Aucune image trouvée dans {image_folder}")
        return
    
    print(f"Trouvé {len(image_paths)} images à traiter.")
    
    # Traiter chaque image
    for img_path in image_paths:
        print(f"\nTraitement de {os.path.basename(img_path)}...")
        predicted_species, predictions, preprocessed_img = predict_single_image(
            model, img_path, species_labels)
        
        if predicted_species is None:
            print(f"Échec du traitement de l'image {img_path}")
            continue
        
        print(f"Prédiction: {predicted_species}")
        
        # Afficher les top 3 prédictions
        top_indices = np.argsort(predictions)[-3:][::-1]
        for i, idx in enumerate(top_indices):
            species_name = species_labels[idx] if idx < len(species_labels) else f"Classe {idx}"
            probability = predictions[idx]
            print(f"  {i+1}. {species_name}: {probability:.4f}")
        
        # Visualiser la prédiction
        visualize_prediction(img_path, predicted_species, predictions, preprocessed_img, species_labels)
        
        # Option pour continuer ou arrêter après chaque image
        if len(image_paths) > 1:
            response = input("Appuyez sur Entrée pour continuer, 'q' pour quitter: ")
            if response.lower() == 'q':
                break


def main():
    parser = argparse.ArgumentParser(
        description='Prédiction des espèces à partir d\'empreintes d\'animaux')
    parser.add_argument('--model', type=str, required=True,
                        help='Chemin vers le modèle entraîné (.pth)')
    parser.add_argument('--image', type=str,
                        help='Chemin vers une image ou un dossier d\'images')
    parser.add_argument('--labels', type=str,
                        help='Chemin vers le fichier contenant les noms d\'espèces')

    args = parser.parse_args()

    # Si l'argument image est un dossier
    if os.path.isdir(args.image):
        test_model_on_folder(args.model, args.image, args.labels)
    # Si c'est un fichier image
    elif os.path.isfile(args.image):
        model, species_labels = load_prediction_model()
        # species_labels = load_species_labels(args.labels)

        # Charger et prétraiter l'image
        img_tensor, img = preprocess_image(args.image)

        # Faire la prédiction
        results = predict_image(model, img_tensor, species_labels)

        # Afficher les résultats
        display_results(img, results)
    else:
        print("Veuillez spécifier une image ou un dossier d'images valide")


if __name__ == "__main__":
    main()
