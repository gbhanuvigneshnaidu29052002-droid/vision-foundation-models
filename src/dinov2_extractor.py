"""
Self-Supervised Feature Representation Extractor using Meta DINOv2
Author: Bhanu Vignesh Naidu Ganeshna
"""

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

class DINOv2FeatureExtractor:
    """
    Extracts deep self-supervised visual representation embeddings using Meta DINOv2.
    """
    def __init__(self, model_name="dinov2_vits14", device="cpu"):
        self.device = device
        print(f"📦 Loading DINOv2 Foundation Model ({model_name})...")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name).to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_embedding(self, image_path):
        """
        Extracts L2-normalized 384-dimensional feature vector.
        """
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(tensor)
            features = F.normalize(features, p=2, dim=1)
            
        return features.cpu().numpy()[0]
