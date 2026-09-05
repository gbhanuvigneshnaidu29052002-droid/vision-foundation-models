"""
Automated Unit Tests for Vision Foundation Models (CLIP & DINOv2)
Author: Bhanu Vignesh Naidu Ganeshna
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import create_demo_image, export_visual_plot


class TestVisionFoundationPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(__file__), "test_artifacts")
        os.makedirs(self.test_dir, exist_ok=True)
        self.demo_img_path = os.path.join(self.test_dir, "test_sample.jpg")

    def tearDown(self):
        if os.path.exists(self.demo_img_path):
            os.remove(self.demo_img_path)
        plot_path = os.path.join(self.test_dir, "clip_zero_shot_probabilities.png")
        if os.path.exists(plot_path):
            os.remove(plot_path)
        if os.path.exists(self.test_dir):
            try:
                os.rmdir(self.test_dir)
            except OSError:
                pass

    def test_demo_image_creation(self):
        """Verify that synthetic test image is created with correct dimensions."""
        create_demo_image(self.demo_img_path)
        self.assertTrue(os.path.exists(self.demo_img_path))
        with Image.open(self.demo_img_path) as img:
            self.assertEqual(img.size, (224, 224))
            self.assertEqual(img.mode, "RGB")

    def test_visual_plot_export(self):
        """Verify that probability bar charts are correctly exported."""
        mock_probs = {
            "crocodile": 0.05,
            "deer": 0.85,
            "gorilla": 0.03,
            "leopard": 0.07,
        }
        export_visual_plot(mock_probs, self.test_dir)
        plot_path = os.path.join(self.test_dir, "clip_zero_shot_probabilities.png")
        self.assertTrue(os.path.exists(plot_path))
        self.assertGreater(os.path.getsize(plot_path), 1000)

    def test_probability_distribution_properties(self):
        """Verify that mock probabilities adhere to simplex probability axioms."""
        probs = np.array([0.05, 0.85, 0.03, 0.07])
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))
        self.assertEqual(int(np.argmax(probs)), 1)


if __name__ == "__main__":
    unittest.main()
