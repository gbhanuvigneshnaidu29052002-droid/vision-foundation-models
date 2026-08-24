"""
Zero-Shot Vision & Foundation Models (CLIP, DINOv2, SAM) Framework
Author: Bhanu Vignesh Naidu Ganeshna
"""

from .clip_zero_shot import CLIPZeroShotClassifier
from .dinov2_extractor import DINOv2FeatureExtractor

__all__ = ["CLIPZeroShotClassifier", "DINOv2FeatureExtractor"]
