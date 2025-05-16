import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
from torchvision.transforms import functional as TF
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import json
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
# Higher resolution captures more detailed textures in footprints
IMG_SIZE = (384, 384)  # or (448, 448) if memory allows
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001
BASE_MODEL = 'resnet'  # 'efficientnet'  # 'mobilenet'
RANDOM_STATE = 42
MODEL_SAVE_PATH = '../models/footprint_classifier.pth'
CLASS_MAPPING_FILE = '../models/class_mapping.json'

# Créer le dossier pour sauvegarder le modèle si nécessaire
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Définition de l'ordre des classes (utilisé pour maintenir un ordre cohérent)
ORDERED_CLASSES = [
    "castor", "cerf_mulet", "cheval", "coyote", "dinde", 
    "ecureuil_gris_occidental", "elephant", "lion", "loutre", "lynx_roux",
    "moufette", "oie", "ours_noir", "rat", "raton_laveur", 
    "renard_gris", "souris", "vison"
]

def get_class_mapping():
    """
    Retourne un mapping des classes dans l'ordre spécifié
    """
    # Créer les mappages de classes dans l'ordre prédéfini
    class_to_idx = {class_name: idx for idx, class_name in enumerate(ORDERED_CLASSES)}
    class_indices = {str(idx): class_name for idx, class_name in enumerate(ORDERED_CLASSES)}
    class_names = ORDERED_CLASSES.copy()
    
    # Sauvegarder le mapping au format JSON pour référence
    mapping = {
        'class_indices': class_indices,
        'class_to_idx': class_to_idx,
        'class_names': class_names
    }
    
    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(CLASS_MAPPING_FILE), exist_ok=True)
    
    # Sauvegarder le mapping au format JSON
    with open(CLASS_MAPPING_FILE, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"Mapping des classes sauvegardé dans {CLASS_MAPPING_FILE}")
    return class_to_idx, class_indices, class_names

# Focal Loss implementation in PyTorch
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.4):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-7

    def forward(self, inputs, targets):
        # Convert targets to one-hot encoding
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Compute focal loss
        probs = torch.softmax(inputs, dim=1)
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)
        
        # Calculate focal weight
        focal_weight = torch.pow(1 - probs, self.gamma)
        
        # Apply class weights (alpha for positive class, 1-alpha for negative)
        pos_weight = targets_one_hot * self.alpha
        neg_weight = (1 - targets_one_hot) * (1 - self.alpha)
        class_weights = pos_weight + neg_weight
        
        # Compute loss
        ce_loss = -torch.log(probs) * targets_one_hot
        loss = focal_weight * class_weights * ce_loss
        
        return loss.sum(dim=1).mean()

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weights
        self.reduction = reduction
        self.eps = 1e-6
        
    def forward(self, inputs, targets):
        # Convert targets to one-hot encoding
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(inputs, dim=1)
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)
        
        # Calculate focal weight
        focal_weight = torch.pow(1 - probs, self.gamma)
        
        # Get probability of the correct class
        batch_size = inputs.size(0)
        pt = probs.gather(1, targets.unsqueeze(1))
        
        # Apply class weights
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = focal_weight * alpha_t.unsqueeze(1)
        
        # Calculate loss
        loss = -focal_weight * torch.log(pt)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class FocalLossWithSmoothing(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, smoothing=0.1):
        super(FocalLossWithSmoothing, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing
        
    def forward(self, inputs, targets):
        # Label smoothing
        num_classes = inputs.size(-1)
        smoothed_labels = torch.zeros_like(inputs)
        smoothed_labels.fill_(self.smoothing / (num_classes - 1))
        smoothed_labels.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        # Focal loss calculation
        log_prob = F.log_softmax(inputs, dim=1)
        prob = torch.exp(log_prob)
        focal_weight = (1 - prob) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            focal_weight = focal_weight * self.alpha.unsqueeze(0).expand_as(focal_weight)
            
        loss = -focal_weight * smoothed_labels * log_prob
        return loss.sum(dim=1).mean()

# Custom dataset class for footprints
class FootprintDataset(Dataset):
    def __init__(self, dataframe, transform=None, class_to_idx=None):
        self.dataframe = dataframe
        self.transform = transform
        
        # Utiliser le mapping de classes fourni ou en créer un à partir du DataFrame
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
            self.classes = sorted(class_to_idx.keys(), key=lambda k: class_to_idx[k])
        else:
            # Cela ne sera utilisé que si aucun mapping n'est fourni
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(dataframe['species'].unique()))}
            self.classes = sorted(dataframe['species'].unique())
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        species = self.dataframe.iloc[idx]['species']
        
        # Vérifier si l'espèce est dans le mapping, sinon donner un avertissement
        if species not in self.class_to_idx:
            print(f"AVERTISSEMENT: Espèce '{species}' non trouvée dans le mapping des classes")
            # Utiliser un indice par défaut ou lever une exception selon le cas
            label = -1  # Indice invalide pour signaler un problème
        else:
            label = self.class_to_idx[species]
            
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


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


def preprocess_data(df, base_dir='../data/Mammiferes_augmented/'):
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

    # Encodage des étiquettes - mais nous allons maintenant utiliser notre mapping prédéfini
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    
    # Vérifier quelles classes sont présentes dans le dataset
    present_classes = df_filtered['species'].unique()
    missing_classes = [cls for cls in ORDERED_CLASSES if cls not in present_classes]
    if missing_classes:
        print(f"AVERTISSEMENT: Certaines classes de la liste prédéfinie ne sont pas dans le dataset: {missing_classes}")
    
    unknown_classes = [cls for cls in present_classes if cls not in ORDERED_CLASSES]
    if unknown_classes:
        print(f"AVERTISSEMENT: Classes inconnues dans le dataset (non dans la liste prédéfinie): {unknown_classes}")
    
    # Nous n'utilisons pas directement le label_encoder pour la transformation
    # mais nous le gardons pour être cohérent avec le code existant
    label_encoder.fit(ORDERED_CLASSES)
    class_names = label_encoder.classes_
    print(f"Classes (dans l'ordre prédéfini): {class_names}")

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


def create_data_loaders(train_df, val_df, test_df):
    """
    Crée des dataloaders avec augmentation
    """
    # Obtenir le mapping des classes dans l'ordre prédéfini
    class_to_idx, _, _ = get_class_mapping()
    
    # Transformations plus sophistiquées pour l'entraînement
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE[0]+40, IMG_SIZE[1]+40)),  # Larger resize for cropping
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomRotation(180),  # Full rotation for footprints
        transforms.RandomAffine(0, translate=(0.4, 0.4), scale=(0.6, 1.5), shear=45),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Simulate different image qualities
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))  # Simulate partial occlusions
    ])

    # Transformations simples pour la validation et le test
    val_test_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Créer les datasets avec le mapping de classes prédéfini
    train_dataset = FootprintDataset(train_df, transform=train_transforms, class_to_idx=class_to_idx)
    val_dataset = FootprintDataset(val_df, transform=val_test_transforms, class_to_idx=class_to_idx)
    test_dataset = FootprintDataset(test_df, transform=val_test_transforms, class_to_idx=class_to_idx)
    
    # Créer les dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=create_balanced_sampler(train_df),
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Conserver le mapping des classes prédéfini
    print("\nUtilisation du mapping de classes prédéfini:")
    for class_name, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        print(f"  {idx}: {class_name}")
    
    return train_loader, val_loader, test_loader, class_to_idx


def build_model(num_classes, base_model_type='efficientnet'):
    print(f"Building model with {base_model_type} as base...")
    
    if base_model_type == 'efficientnet':
        # EfficientNet captures texture details better
        model = models.efficientnet_b3(pretrained=True)
        num_ftrs = model.classifier[1].in_features  # Access the in_features of the Linear layer

        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 1536),
            nn.BatchNorm1d(1536),
            nn.SiLU(inplace=True),  # SiLU/Swish activation
            nn.Dropout(0.4),
            nn.Linear(1536, 768),
            nn.BatchNorm1d(768),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(768, num_classes)
        )
    elif base_model_type == 'convnext':
        # ConvNext has better feature extraction
        model = models.convnext_small(pretrained=True)
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, num_classes)
        
    return model


def calculate_class_weights(train_df):
    """
    Calcule les poids des classes en fonction de leur fréquence
    Les classes moins représentées reçoivent un poids plus élevé
    """
    # Compter le nombre d'échantillons par classe
    class_counts = train_df['species'].value_counts()
    total_samples = len(train_df)
    n_classes = len(class_counts)

    # Calculer les poids: inversement proportionnels à la fréquence de la classe
    weights = []
    for cls_name in sorted(class_counts.index):
        count = class_counts[cls_name]
        weight = total_samples / (n_classes * count)
        weights.append(weight)
        print(f"- {cls_name}: {weight:.4f}")

    return torch.FloatTensor(weights).to(device)


def train_model(model, train_loader, val_loader, class_weights=None, patience=15):
    """Improved training with better scheduling"""
    model = model.to(device)
    
    # Layer-wise learning rates
    params = [
        {'params': get_backbone_params(model), 'lr': LEARNING_RATE * 0.1},
        {'params': get_head_params(model), 'lr': LEARNING_RATE}
    ]
    
    # AdamW optimizer with weight decay
    optimizer = optim.AdamW(params, lr=LEARNING_RATE, weight_decay=0.01)
    
    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    # Use focal loss for imbalanced dataset if class weights are provided
    if class_weights is not None:
        criterion = WeightedFocalLoss(alpha=class_weights, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Early stopping setup
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Mixup augmentation
            if np.random.random() > 0.5:  # Apply Mixup 50% of the time
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.2)
                inputs, targets_a, targets_b = inputs.to(device), targets_a.to(device), targets_b.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Compute loss with Mixup
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                # Forward pass without Mixup
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': train_loss/train_total, 
                'acc': 100.*train_correct/train_total
            })
        
        # Calculate epoch statistics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Update metrics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Calculate validation statistics
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Step the scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{EPOCHS}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print("Training completed")
    return model, history


def evaluate_model(model, test_loader, class_names, adjust=True):
    """
    Évalue le modèle sur l'ensemble de test
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    test_loss = 0
    test_correct = 0
    
    # Utiliser la même fonction de perte que pour l'entraînement
    criterion = nn.CrossEntropyLoss()
    
    print("\nÉvaluation du modèle sur l'ensemble de test...")
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Évaluation"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Calculer la perte
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            
            # Prédictions
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == targets).sum().item()
            
            # Collecter les prédictions et les cibles pour les métriques
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculer la perte et la précision moyennes
    test_loss /= len(test_loader.dataset)
    test_accuracy = test_correct / len(test_loader.dataset)
    
    print(f"Perte sur l'ensemble de test: {test_loss:.4f}")
    print(f"Précision sur l'ensemble de test: {test_accuracy:.4f}")
    
    # Rapport de classification
    print("\nRapport de classification:")
    print(classification_report(all_targets, all_predictions, target_names=class_names))
    
    return all_targets, all_predictions


def plot_training_curves(history):
    """
    Affiche les courbes d'entraînement (précision et perte)
    """
    # Créer une figure avec deux sous-graphiques
    plt.figure(figsize=(12, 5))

    # Graphique de précision
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Entraînement')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Précision du modèle')
    plt.xlabel('Epoch')
    plt.ylabel('Précision')
    plt.legend()

    # Graphique de perte
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Entraînement')
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

    # Créer les transformations pour l'augmentation
    augmentation_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomRotation(45),
        transforms.RandomAffine(0, translate=(0.35, 0.35), scale=(0.65, 1.35), shear=35),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35)
    ])

    # Visualiser les augmentations
    plt.figure(figsize=(15, 5 * len(sample_images)))

    for i, img_path in enumerate(sample_images):
        # Charger l'image
        img = Image.open(img_path).convert('RGB')

        # Afficher l'image originale
        plt.subplot(len(sample_images), 4, i*4 + 1)
        plt.imshow(img)
        plt.title(f"Original: {os.path.basename(img_path)}")
        plt.axis('off')

        # Appliquer différentes augmentations
        for j in range(3):
            aug_img = augmentation_transforms(img)
            plt.subplot(len(sample_images), 4, i*4 + j + 2)
            plt.imshow(aug_img)
            plt.title(f"Augmentation {j+1}")
            plt.axis('off')

    plt.tight_layout()
    plt.savefig('augmentation_examples.png')
    plt.show()


def mixup_data(x, y, alpha=0.2):
    """
    Applies Mixup augmentation to the batch.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the Mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def tta_predict(model, image, num_augments=10):
    """
    Test-time augmentation to improve prediction robustness
    """
    model.eval()
    
    # Create different augmentations
    tta_transforms = [
        transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Add more transformations as needed
    ]
    
    # Collect predictions from all augmentations
    all_preds = []
    with torch.no_grad():
        # Original image prediction
        for transform in tta_transforms:
            img_tensor = transform(image).unsqueeze(0).to(device)
            output = model(img_tensor)
            all_preds.append(output)
    
    # Average predictions
    final_pred = torch.mean(torch.stack(all_preds), dim=0)
    return final_pred


def create_balanced_sampler(train_df):
    """Create a weighted sampler to balance classes during training"""
    class_counts = train_df['species'].value_counts().to_dict()
    weights = []
    
    for species in train_df['species']:
        weight = 1.0 / class_counts[species]
        weights.append(weight)
        
    weights = torch.FloatTensor(weights)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    return sampler


def predict_with_tta(model, image, num_augs=5):
    """Improve prediction accuracy with test-time augmentation"""
    model.eval()
    
    # Create different test-time augmentations
    tta_transforms = [
        transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomRotation((90, 90)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    ]
    
    # Collect predictions
    all_preds = []
    with torch.no_grad():
        for transform in tta_transforms:
            img_tensor = transform(image).unsqueeze(0).to(device)
            output = model(img_tensor)
            all_preds.append(output)
    
    # Average predictions
    final_pred = torch.mean(torch.stack(all_preds), dim=0)
    return final_pred


def get_backbone_params(model):
    """
    Returns the parameters of the backbone layers.
    """
    # Adjust this based on your model architecture
    if BASE_MODEL == 'efficientnet':
        return model.features.parameters()
    elif BASE_MODEL == 'resnet':
        # Return parameters, not modules
        backbone = list(model.children())[:-2]
        return [p for m in backbone for p in m.parameters()]
    elif BASE_MODEL == 'convnext':
        return model.features.parameters()
    else:
        raise ValueError(f"Unsupported base model: {BASE_MODEL}")


def get_head_params(model):
    """
    Returns the parameters of the classification head.
    """
    if BASE_MODEL == 'efficientnet':
        return model.classifier.parameters()
    elif BASE_MODEL == 'resnet':
        # Return parameters, not modules
        head = list(model.children())[-2:]
        return [p for m in head for p in m.parameters()]
    elif BASE_MODEL == 'convnext':
        return model.classifier.parameters()
    else:
        raise ValueError(f"Unsupported base model: {BASE_MODEL}")


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

    # 3. Création des dataloaders avec le mapping de classes prédéfini
    train_loader, val_loader, test_loader, class_to_idx = create_data_loaders(
        train_df, val_df, test_df)

    # 4. Calcul des poids des classes pour gérer le déséquilibre
    print("\nPoids des classes:")
    class_weights = calculate_class_weights(train_df)

    # 5. Construction du modèle
    model = build_model(len(class_names))
    model = model.to(device)

    # Phase 1: Entraîner uniquement les couches supérieures
    print("\n--- Phase 1: Entraînement des couches supérieures ---")

    # Geler complètement le modèle de base (sauf les couches finales)
    # Pour un modèle ResNet, nous gardons la dernière couche FC modifiable
    if BASE_MODEL == 'resnet':
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
    
    # Optimiseur pour la phase 1
    optimizer_phase1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    # Entraîner la phase 1
    model_phase1, history_phase1 = train_model(
        model, 
        train_loader, 
        val_loader, 
        class_weights=class_weights, 
        patience=5
    )
    
    # Sauvegarder le modèle de la phase 1
    phase1_save_path = MODEL_SAVE_PATH.replace('.pth', '_phase1.pth')
    torch.save(model_phase1.state_dict(), phase1_save_path)
    print(f"Modèle phase 1 sauvegardé à {phase1_save_path}")

    # Phase 2: Fine-tuning du modèle complet
    print("\n--- Phase 2: Fine-tuning du modèle complet ---")

    # Pour ResNet, dégeler progressivement les couches
    if BASE_MODEL == 'resnet':
        # Dégeler les derniers blocs (les 25% dernières couches)
        total_layers = len(list(model_phase1.parameters()))
        freeze_cutoff = int(total_layers * 0.75)
        i = 0
        for param in model_phase1.parameters():
            if i >= freeze_cutoff:
                param.requires_grad = True
            i += 1
    
    # Optimiseur pour la phase 2 avec un taux d'apprentissage plus faible
    optimizer_phase2 = optim.Adam(filter(lambda p: p.requires_grad, model_phase1.parameters()), lr=0.00005)
    
    # Entraîner la phase 2
    final_model, history_phase2 = train_model(
        model_phase1, 
        train_loader, 
        val_loader, 
        class_weights=class_weights, 
        patience=10
    )
    
    # Combiner les historiques
    history = {}
    for k in history_phase1:
        if k in history_phase2:
            history[k] = history_phase1[k] + history_phase2[k]
    
    # 7. Évaluation du modèle
    y_true, y_pred = evaluate_model(final_model, test_loader, class_names)
    
    # 8. Visualisation des résultats
    plot_training_curves(history)
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    # 9. Sauvegarde du mapping des classes
    # Utiliser nos mappings prédéfinis
    class_to_idx, class_indices, class_names_list = get_class_mapping()
    
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'class_indices': class_indices,
        'class_to_idx': class_to_idx,
        'class_names': class_names_list,
        'label_encoder': label_encoder
    }, MODEL_SAVE_PATH)
    print(f"Modèle et indices de classes sauvegardés dans {MODEL_SAVE_PATH}")
    
    # Afficher explicitement l'ordre des classes pour vérification
    print("\nOrdre des classes sauvegardé:")
    for idx, cls_name in enumerate(class_names_list):
        print(f"{idx}: {cls_name} (indice dans le modèle: {class_to_idx.get(cls_name, 'N/A')})")
    
    # Visualiser les augmentations
    visualize_augmentations(
        train_df, ['Raton laveur', 'Renard', 'Lynx', 'Écureuil'])
    
    print("\nEntraînement terminé avec succès!")

    # Log des résultats finaux
    log_file = "training_log.txt"
    with open(log_file, "w") as f:
        f.write("Training Results:\n\n")
        f.write("Configuration:\n")
        f.write(f"  IMG_SIZE: {IMG_SIZE}\n")
        f.write(f"  BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"  EPOCHS: {EPOCHS}\n")
        f.write(f"  LEARNING_RATE: {LEARNING_RATE}\n")
        f.write(f"  BASE_MODEL: {BASE_MODEL}\n")
        f.write("\n")

        f.write("Class Mapping:\n")
        for idx, cls_name in enumerate(class_names_list):
            f.write(f"  {idx}: {cls_name} (indice: {class_to_idx.get(cls_name, 'N/A')})\n")
        f.write("\n")

        f.write("Training History (Phase 1):\n")
        for k, v in history_phase1.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("Training History (Phase 2):\n")
        for k, v in history_phase2.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("Evaluation Metrics:\n")
        report = classification_report(y_true, y_pred, target_names=class_names)
        f.write(report)
        f.write("\n")

        print(f"Les résultats ont été enregistrés dans {log_file}")


if __name__ == "__main__":
    # Définir les configurations de seed pour la reproductibilité
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)
    
    # Exécuter le pipeline principal
    main()
