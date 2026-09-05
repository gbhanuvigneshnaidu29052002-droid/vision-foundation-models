---
name: Bug Report
about: Report an issue with model loading, tensor dimension mismatch, or execution failure
title: "[BUG] "
labels: ["bug"]
assignees: ""
---

## Description
A clear and concise description of the bug.

## Steps to Reproduce
1. Command executed: `python3 main.py --image ...`
2. Arguments or code snippet used:
3. Error observed.

## Expected Behavior
What was expected to occur (e.g. valid classification probabilities or normalized 384-d embedding vector).

## Actual Behavior
What actually happened (e.g., CUDA OOM exception, tokenizer error, tensor shape mismatch).

## Console Logs & Stack Traces
```text
Paste logs or terminal output here
```

## Environment Details
- **OS**: Ubuntu 22.04 / Linux / macOS / Windows
- **Python**: 3.10.x
- **PyTorch**: 2.x
- **Transformers**: 4.x / 5.x
- **Device**: CPU / CUDA / MPS (specify GPU model if applicable)

## Additional Context
Add any other context about the problem here.
