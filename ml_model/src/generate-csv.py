# Créer un modèle de CSV simplifié avec uniquement image_path et species
import pandas as pd
import os
import glob

# Chemin vers le dossier contenant les images d'empreintes par espèce
base_path = 'data/Mammiferes_augmented/'

# Liste des espèces basée sur les dossiers existants
species_folders = [d for d in os.listdir(
    base_path) if os.path.isdir(os.path.join(base_path, d))]
print(f"Espèces trouvées: {species_folders}")

# Liste pour stocker les données
data_rows = []

# Parcourir chaque dossier d'espèces
for species in species_folders:
    species_path = os.path.join(base_path, species)

    # Trouver toutes les images dans ce dossier (supposons des jpg, png, jpeg)
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(os.path.join(species_path, ext)))

    # Pour chaque image trouvée, créer une entrée dans notre dataset
    for img_path in image_paths:
        # Extraire juste le chemin relatif
        rel_path = os.path.relpath(img_path, start=os.path.dirname(base_path))

        # Ajouter seulement le chemin d'image et l'espèce
        row = {
            'image_path': rel_path,
            'species': species,
        }
        data_rows.append(row)

# Créer le DataFrame
if data_rows:
    footprint_data = pd.DataFrame(data_rows)

    # Enregistrer en CSV
    csv_path = 'data/footprint_dataset_augmented.csv'
    footprint_data.to_csv(csv_path, index=False)

    # Afficher les premières lignes
    print(f"Dataset créé avec {len(footprint_data)} images d'empreintes")
    print(f"Dataset enregistré sous: {os.path.abspath(csv_path)}")

    # Aperçu des données
    footprint_data.head()
else:
    print("Aucune image trouvée dans les dossiers d'espèces.")
