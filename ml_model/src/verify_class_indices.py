import torch
import os
import argparse
from PIL import Image
import torchvision.transforms as transforms
from model import FootprintClassifier

def load_model_and_classes(model_path):
    """
    Load the model and class mapping information from the saved file
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Extract class mapping information
    class_indices = checkpoint.get('class_indices', {})
    class_to_idx = checkpoint.get('class_to_idx', {})
    class_names = checkpoint.get('class_names', [])
    
    print("\n=== Class Mapping Information ===")
    print(f"Number of classes: {len(class_indices)}")
    print("\nclass_indices (index → class name):")
    for idx, name in sorted(class_indices.items()):
        print(f"  {idx}: {name}")
    
    print("\nclass_to_idx (class name → index):")
    for name, idx in sorted(class_to_idx.items()):
        print(f"  {name}: {idx}")
    
    if class_names:
        print("\nOrdered class_names:")
        for i, name in enumerate(class_names):
            print(f"  {i}: {name}")
    
    return checkpoint, class_indices

def predict_single_image(model_path, image_path):
    """
    Make a prediction for a single image and display the results
    """
    checkpoint, class_indices = load_model_and_classes(model_path)
    
    # Load model
    num_classes = len(class_indices)
    model = FootprintClassifier(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Prepare the image
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Match the training size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Display top 5 predictions
    top_probs, top_indices = torch.topk(probabilities, 5)
    
    print(f"\n=== Prediction for {os.path.basename(image_path)} ===")
    for i in range(5):
        idx = top_indices[i].item()
        prob = top_probs[i].item()
        class_name = class_indices.get(str(idx), f"Unknown ({idx})")
        print(f"{i+1}. {class_name}: {prob:.4f} ({idx})")
        
    return class_indices, top_indices, top_probs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify class indices and test model prediction")
    parser.add_argument("--model", default="../models/footprint_classifier.pth", 
                        help="Path to the model file")
    parser.add_argument("--image", required=False,
                        help="Path to an image file for prediction testing")
    
    args = parser.parse_args()
    
    # Always load and display class indices
    checkpoint, class_indices = load_model_and_classes(args.model)
    
    # If an image is provided, test prediction
    if args.image:
        predict_single_image(args.model, args.image)
