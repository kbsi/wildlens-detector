import torch
from PIL import Image
import argparse
import os
import json
from torchvision import transforms
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder
from model import FootprintClassifier
import matplotlib.pyplot as plt  # Add this import for plotting

# Define ORDERED_CLASSES in prediction_fix.py to match train_new.py
ORDERED_CLASSES = [
    "castor", "cerf_mulet", "cheval", "coyote", "dinde", 
    "ecureuil_gris_occidental", "elephant", "lion", "loutre", "lynx_roux",
    "moufette", "oie", "ours_noir", "rat", "raton_laveur", 
    "renard_gris", "souris", "vison"
]

# Define your model architecture here.  This is a placeholder!
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=len(ORDERED_CLASSES)):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 74 * 74, num_classes)  # Adjusted for 299x299 input
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def load_model(model_path):
    """
    Charge le modèle et les noms de classes.
    """
    try:
        # Determine the device to use
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate the model
        model = FootprintClassifier().to(device)

        # Load the model, explicitly allowing unsafe objects
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Check if the checkpoint is a state_dict or the entire model
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        else:
            model_state_dict = checkpoint  # Assume it's directly the state_dict

        # Load the state dict into the model
        model.load_state_dict(model_state_dict)

        # Ensure the model is on the correct device AFTER loading the state dict
        model = model.to(device)

        model.eval()  # Set the model to evaluation mode

        #Recreate class names and indices from ORDERED_CLASSES
        class_names = ORDERED_CLASSES
        class_indices = {str(i): class_name for i, class_name in enumerate(ORDERED_CLASSES)}

        print("Modèle chargé avec succès.")
        return model, class_names, class_indices
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        raise

def predict_image(model, image_path, class_names, class_indices):
    """
    Prédire la classe d'une image et afficher les résultats
    """
    # Transformation identique à celle utilisée pendant l'entraînement
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Charger et transformer l'image
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
    except Exception as e:
        print(f"Erreur lors du chargement de l'image {image_path}: {e}")
        return None
    
    # Move the input tensor to the same device as the model
    device = next(model.parameters()).device  # Get the device of the model's parameters
    input_tensor = input_tensor.to(device)
    
    # Faire la prédiction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Obtenir les 5 meilleures prédictions
    top_k = min(5, len(probabilities))
    top_probs, top_indices = torch.topk(probabilities, top_k)
    
    # Afficher les résultats
    print(f"\nPrédiction pour {os.path.basename(image_path)}:")
    predicted_classes = []
    
    for i in range(len(top_indices)):
        idx = top_indices[i].item()
        # Utilisation de str(idx) pour accéder au dictionnaire class_indices
        if (str(idx) in class_indices):
            class_name = class_indices[str(idx)]
        elif idx < len(class_names):
            class_name = class_names[idx]
        else:
            class_name = f"Classe inconnue ({idx})"
            
        probability = top_probs[i].item()
        predicted_classes.append((class_name, probability))
        print(f"{class_name}: {probability:.4f}")
    
    # Visualiser l'image avec la prédiction
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f"Prédiction: {predicted_classes[0][0]} ({predicted_classes[0][1]:.2f})")
    plt.axis('off')
    plt.show()
    
    return predicted_classes

def debug_model_classes(model_path):
    """
    Fonction de débogage pour examiner en détail les mappages de classes
    """
    print("\n=== DÉBOGAGE DES MAPPAGES DE CLASSES ===")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Examiner le contenu du checkpoint
    print("Clés disponibles dans le checkpoint:", list(checkpoint.keys()))
    
    # Examiner class_indices
    class_indices = checkpoint.get('class_indices', {})
    print("\nclass_indices:", type(class_indices))
    if isinstance(class_indices, dict):
        for k, v in sorted(class_indices.items(), key=lambda x: x[0]):
            print(f"  '{k}': '{v}'")
    else:
        print("  Non disponible ou format inattendu")
    
    # Examiner class_to_idx
    class_to_idx = checkpoint.get('class_to_idx', {})
    print("\nclass_to_idx:", type(class_to_idx))
    if isinstance(class_to_idx, dict):
        for k, v in sorted(class_to_idx.items()):
            print(f"  '{k}': {v}")
    else:
        print("  Non disponible ou format inattendu")
    
    # Examiner class_names
    class_names = checkpoint.get('class_names', [])
    print("\nclass_names:", type(class_names))
    if isinstance(class_names, list):
        for i, name in enumerate(class_names):
            print(f"  {i}: '{name}'")
    else:
        print("  Non disponible ou format inattendu")
    
    return checkpoint

def main():
    parser = argparse.ArgumentParser(description="Prédire la classe d'une empreinte d'animal")
    parser.add_argument("--model", default="../models/footprint_classifier.pth", 
                        help="Chemin vers le fichier du modèle entraîné")
    parser.add_argument("--image", required=True, 
                        help="Chemin vers l'image à prédire")
    parser.add_argument("--debug", action="store_true", 
                        help="Active le mode de débogage pour examiner les mappages de classes")
    args = parser.parse_args()
    
    # Option de débogage
    if args.debug:
        debug_model_classes(args.model)
    
    # Charger le modèle et les informations sur les classes
    model, class_names, class_indices = load_model(args.model)
    
    # Afficher les classes disponibles
    if class_names:
        print("\nClasses disponibles:")
        for i, name in enumerate(class_names):
            print(f"  {i}: {name}")
    
    # Prédire l'image
    predict_image(model, args.image, class_names, class_indices)

if __name__ == "__main__":
    main()
