## Summary

Please summarize the changes introduced in this pull request and the motivation behind them.

Fixes # (issue)

---

## Type of Change

- [ ] **Bug fix** (non-breaking fix resolving an issue or error)
- [ ] **New feature** (adds a new foundation model, downstream task, or evaluation metric)
- [ ] **Performance optimization** (faster inference latency, memory reduction, quantization)
- [ ] **Visualization / Reporting** (new plots, benchmark tables, analysis)
- [ ] **Documentation / Tests** (README enhancements, test cases, setup guides)

---

## Testing & Verification

Please confirm that the following verification steps have been executed:

- [ ] Dependencies installed via `pip install -r requirements.txt`
- [ ] Automated unit test suite passes: `python3 -m unittest discover -s tests -v`
- [ ] Tested CLI entrypoint: `python3 main.py --output_dir results`
- [ ] Tested on target hardware (CPU / CUDA GPU / Apple Silicon MPS)
- [ ] Exported benchmark artifacts verified in `results/`

---

## Checklist

- [ ] My code conforms to PEP 8 style standards.
- [ ] I have added appropriate docstrings and comments for complex logic.
- [ ] No large model weights (`.pt`, `.bin`, `.safetensors`) or temporary files are committed.
- [ ] I have updated the documentation or README if applicable.
