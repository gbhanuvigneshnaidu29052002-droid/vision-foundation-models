# Contributing to Zero-Shot Vision-Language & Self-Supervised Foundation Models

Thank you for your interest in contributing to **Zero-Shot Vision-Language & Self-Supervised Foundation Models (CLIP & DINOv2)**! We welcome contributions from machine learning practitioners, computer vision researchers, and open-source developers.

Please review the following guidelines before submitting issues or pull requests.

---

## Code of Conduct

All contributors and participants are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please report any unacceptable behavior to the project maintainer.

---

## Areas of Contribution

We encourage contributions in the following domains:

- **Model Zoo Expansion**: Integrate modern foundation vision models such as SigLIP, BLIP-2, Meta SAM (Segment Anything), or GroundingDINO.
- **Prompt Engineering & Ensembling**: Implement template ensembling (e.g., ImageNet 80-prompt ensemble) to improve zero-shot classification robustness.
- **Downstream Tasks**: Add zero-shot image-text retrieval, cross-modal semantic search, k-NN feature evaluation on benchmark datasets (CIFAR-100, STL-10, ImageNet-1k).
- **Optimization & Export**: Add ONNX / TensorRT export scripts, INT8/FP16 quantization, or optimized inference pipelines with `torch.compile`.
- **Visualization & Metrics**: Enhance evaluation metrics (Top-1, Top-5, Mean Reciprocal Rank, t-SNE / UMAP clustering visualizations).

---

## Reporting Issues & Bugs

Before filing a new issue, please check existing [GitHub Issues](https://github.com/gbhanuvigneshnaidu29052002-droid/vision-foundation-models/issues) to avoid duplicates.

When reporting a bug:
1. Use our [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md).
2. Specify your hardware and environment details:
   - OS: Ubuntu 22.04 / Linux / macOS / Windows
   - Python Version: Python 3.10+
   - PyTorch & Transformers versions
   - Device: CPU, CUDA (with GPU model), or Apple Silicon MPS
3. Provide a minimal code snippet to reproduce the error.
4. Include full error stack traces and any relevant logs.

---

## Development Workflow

### 1. Fork & Clone Repository
```bash
git clone https://github.com/<your-username>/vision-foundation-models.git
cd vision-foundation-models
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Create a Feature Branch
```bash
git checkout -b feature/siglip-zero-shot-support
# or
git checkout -b fix/cuda-fp16-normalization-bug
```

### 4. Run Automated Unit Tests
Verify that all unit tests pass:
```bash
python3 -m unittest discover -s tests -v
```

### 5. Run Benchmark CLI
Verify end-to-end inference and visual plot generation:
```bash
python3 main.py --output_dir results
```

---

## Code Quality Standards

- **Formatting**: Adhere to PEP 8 standards. Use `black` and `flake8` if available.
- **Type Annotations**: Add type hints where appropriate for public methods and functions.
- **Device Agnostic**: Ensure code runs on both `cpu` and `cuda` devices without hardcoding GPU device IDs.
- **Git Hygiene**: Keep commits atomic and informative. Never commit large model checkpoint weights (`.pt`, `.bin`, `.safetensors`), cached virtual environments, or `__pycache__` directories.

---

## Submitting a Pull Request

1. Fill out the [Pull Request Template](.github/pull_request_template.md).
2. Ensure all unit tests pass (`python3 -m unittest discover -s tests -v`).
3. If introducing a new model or feature, include benchmark numbers or generated visual output in the PR description.

Thank you for contributing to open foundation model research!
