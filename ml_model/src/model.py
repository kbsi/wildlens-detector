import torch
import torch.nn as nn
import torchvision.models as models

class FootprintClassifier(nn.Module):
    def __init__(self, num_classes=18, pretrained=False):  # Adjust num_classes to the actual number of classes
        super(FootprintClassifier, self).__init__()
        # Instead of wrapping a ResNet with 'self.resnet = models.resnet50()',
        # we'll directly inherit from the ResNet50 architecture
        # This way the state_dict keys will match the loaded model
        
        # Create a base ResNet model
        base_model = models.resnet50(pretrained=pretrained)
        
        # Copy all the attributes from ResNet
        for attr_name in dir(base_model):
            if not attr_name.startswith('__') and not callable(getattr(base_model, attr_name)):
                setattr(self, attr_name, getattr(base_model, attr_name))
                
        # Copy all modules
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        
        # The saved model's FC layer has different sizes than we initially assumed
        # According to the error message, the FC layer sizes should be:
        # fc.0: Linear(2048, 1024)
        # fc.3: Linear(1024, 512)
        # fc.6: Linear(512, 18)
        num_ftrs = base_model.fc.in_features  # This should be 2048 for ResNet50
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),  # fc.0 - Corrected size
            nn.ReLU(),                  # fc.1
            nn.Dropout(0.2),            # fc.2
            nn.Linear(1024, 512),       # fc.3 - Corrected size
            nn.ReLU(),                  # fc.4
            nn.Dropout(0.2),            # fc.5
            nn.Linear(512, num_classes)  # fc.6 - This was already correct if num_classes=18
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

if __name__ == "__main__":
    # Example usage
    model = FootprintClassifier()
    print(model)

    # Example input
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224 image
    output = model(dummy_input)
    print("Output shape:", output.shape)