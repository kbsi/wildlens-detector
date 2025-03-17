import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from ml_model.src.model import MyModel  # Adjust the import based on your model's definition

class FootprintPredictor:
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Adjust size based on your model's input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_model(self, model_path: str):
        model = MyModel()  # Initialize your model architecture
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image.unsqueeze(0)  # Add batch dimension

    def predict(self, image_path: str) -> dict:
        image_tensor = self.preprocess_image(image_path)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        return {
            'predicted_class': predicted_class.item(),
            'confidence': confidence.item()
        }

if __name__ == "__main__":
    model_path = os.path.join("ml_model", "models", "best_model.pth")  # Adjust path as necessary
    predictor = FootprintPredictor(model_path)
    result = predictor.predict("path_to_your_image.jpg")  # Replace with actual image path
    print(result)