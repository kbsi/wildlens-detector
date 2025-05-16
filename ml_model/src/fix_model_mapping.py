import torch
import os
import argparse
import json
from pathlib import Path

def fix_class_mapping(model_path, output_path=None):
    """
    Répare les mappages de classes dans un modèle sauvegardé
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fichier modèle non trouvé: {model_path}")
    
    # Charger le checkpoint
    print(f"Chargement du modèle depuis {model_path}...")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Extraire les mappages existants
    class_indices = checkpoint.get('class_indices', {})
    class_to_idx = checkpoint.get('class_to_idx', {})
    class_names = checkpoint.get('class_names', [])
    
    print("\nMappages actuels:")
    print(f"- class_indices: {len(class_indices)} entrées")
    print(f"- class_to_idx: {len(class_to_idx)} entrées") 
    print(f"- class_names: {len(class_names)} entrées")
    
    # Vérifier si les mappages sont cohérents
    if class_indices and class_to_idx and class_names:
        print("Tous les mappages de classes sont présents.")
    else:
        print("Certains mappages manquent ou sont incomplets, tentative de reconstruction...")
    
    # Reconstruction des mappages
    updated = False
    
    # Si class_indices existe mais les clés ne sont pas des chaînes
    if class_indices:
        new_class_indices = {}
        for k, v in class_indices.items():
            if not isinstance(k, str):
                new_class_indices[str(k)] = v
                updated = True
            else:
                new_class_indices[k] = v
        
        if updated:
            print("class_indices mis à jour: conversion des clés en chaînes")
            class_indices = new_class_indices
            checkpoint['class_indices'] = class_indices
    
    # Si class_to_idx n'existe pas mais class_indices existe
    if not class_to_idx and class_indices:
        class_to_idx = {v: int(k) for k, v in class_indices.items()}
        checkpoint['class_to_idx'] = class_to_idx
        print("class_to_idx reconstruit à partir de class_indices")
        updated = True
    
    # Si class_names n'existe pas mais class_to_idx existe
    if not class_names and class_to_idx:
        class_names = sorted(class_to_idx.keys(), key=lambda x: class_to_idx[x])
        checkpoint['class_names'] = class_names
        print("class_names reconstruit à partir de class_to_idx")
        updated = True
    
    # Afficher les détails des classes
    print("\nDétails des classes reconstruites:")
    for i, name in enumerate(class_names):
        idx = class_to_idx.get(name, "?")
        idx_from_indices = next((k for k, v in class_indices.items() if v == name), "?")
        print(f"  {i}: {name} (class_to_idx: {idx}, class_indices key: {idx_from_indices})")
    
    # Sauvegarder si des mises à jour ont été effectuées
    if updated:
        if output_path is None:
            # Générer un nom de fichier par défaut si non spécifié
            base_path = Path(model_path)
            output_path = str(base_path.parent / (base_path.stem + "_fixed" + base_path.suffix))
        
        print(f"\nSauvegarde du modèle réparé vers {output_path}...")
        torch.save(checkpoint, output_path)
        print("Modèle sauvegardé avec succès!")
        
        # Sauvegarder aussi les mappages au format JSON pour référence
        json_path = Path(output_path).with_suffix('.json')
        with open(json_path, 'w') as f:
            json_data = {
                'class_indices': {k: v for k, v in class_indices.items()},
                'class_to_idx': {k: v for k, v in class_to_idx.items()},
                'class_names': class_names
            }
            json.dump(json_data, f, indent=2)
        print(f"Mappages exportés vers {json_path}")
    else:
        print("Aucune mise à jour nécessaire.")
    
    return checkpoint, class_indices, class_to_idx, class_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Réparer les mappages de classes dans un modèle sauvegardé")
    parser.add_argument("--model", required=True, help="Chemin vers le fichier modèle à réparer")
    parser.add_argument("--output", help="Chemin où sauvegarder le modèle réparé (optionnel)")
    
    args = parser.parse_args()
    fix_class_mapping(args.model, args.output)
