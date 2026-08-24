from setuptools import setup, find_packages

setup(
    name="foundation_models_vision",
    version="1.0.0",
    description="Zero-Shot Computer Vision Suite leveraging CLIP, DINOv2, and SAM Foundation Models",
    author="Bhanu Vignesh Naidu Ganeshna",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "transformers>=4.20.0",
        "Pillow>=8.0.0"
    ],
    entry_points={
        'console_scripts': [
            'foundation-vision=main:main',
        ],
    },
    python_requires='>=3.8',
)
