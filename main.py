"""
Main CLI Entrypoint for Vision Foundation Models Benchmark (CLIP & DINOv2)
Author: Bhanu Vignesh Naidu Ganeshna
"""

import os
import sys
import argparse
import json
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.clip_zero_shot import CLIPZeroShotClassifier
from src.dinov2_extractor import DINOv2FeatureExtractor

def create_demo_image(path):
    img = Image.new("RGB", (224, 224), color=(200, 50, 50))
    img.save(path)

def export_visual_plot(probs, output_dir):
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)
    labels = list(probs.keys())
    values = [probs[l] * 100 for l in labels]
    
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, values, color=['#3498DB', '#2ECC71', '#9B59B6', '#E67E22'], edgecolor='black')
    plt.ylabel('Zero-Shot Probability (%)', fontsize=11)
    plt.title('CLIP Zero-Shot Vision-Language Classification', fontsize=12, fontweight='bold')
    plt.ylim([0, 100])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1.5, f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clip_zero_shot_probabilities.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Foundation Vision Models Benchmark")
    parser.add_argument("--image", type=str, default="data/demo_image.jpg")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.image), exist_ok=True)
    
    if not os.path.exists(args.image):
        create_demo_image(args.image)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Device initialized: {device}")
    
    print("\n--- 1. CLIP Zero-Shot Classification ---")
    candidate_labels = ["crocodile", "deer", "gorilla", "leopard"]
    clip = CLIPZeroShotClassifier(device=device)
    top_label, probs = clip.classify(args.image, candidate_labels)
    
    print(f"🎯 CLIP Predicted Class: {top_label}")
    for k, v in probs.items():
        print(f"   - {k:<12}: {v*100:.2f}%")
        
    export_visual_plot(probs, args.output_dir)
        
    print("\n--- 2. DINOv2 Feature Representation Extraction ---")
    dinov2 = DINOv2FeatureExtractor(device=device)
    embedding = dinov2.extract_embedding(args.image)
    print(f"✅ DINOv2 Embedding Extracted! Vector Shape: {embedding.shape}, L2 Norm: {np.linalg.norm(embedding):.4f}")
    
    results = {
        'CLIP_Top_Prediction': top_label,
        'CLIP_Class_Probabilities': probs,
        'DINOv2_Embedding_Dim': int(embedding.shape[0])
    }
    
    res_file = os.path.join(args.output_dir, "foundation_benchmark.json")
    with open(res_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n✅ Foundation Model Benchmark exported to {res_file}")

if __name__ == "__main__":
    import torch
    import numpy as np
    main()
