"""
Zero-Shot Vision-Language Classification using OpenAI CLIP
Author: Bhanu Vignesh Naidu Ganeshna
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPZeroShotClassifier:
    """
    Zero-shot image classifier leveraging vision-language alignment in CLIP.
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cpu"):
        self.device = device
        print(f"📦 Loading CLIP Foundation Model ({model_name})...")
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def classify(self, image_path, candidate_labels, prompt_template="a photo of a {}"):
        """
        Classifies an image against arbitrary text candidate labels without training.
        """
        image = Image.open(image_path).convert("RGB")
        text_prompts = [prompt_template.format(label) for label in candidate_labels]
        
        inputs = self.processor(text=text_prompts, images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
            
        results = {label: float(prob) for label, prob in zip(candidate_labels, probs)}
        top_label = candidate_labels[probs.argmax()]
        return top_label, results
