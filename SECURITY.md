# Security Policy

## Supported Versions

Security updates and vulnerability patches are maintained for the following environments:

| Version / Branch | Python Version | PyTorch | Transformers | Status |
| :--- | :--- | :--- | :--- | :--- |
| `main` | 3.10+ | 2.0+ | 4.20+ | :white_check_mark: Supported |
| Older commits | < 3.10 | < 2.0 | < 4.20 | :x: Not supported |

---

## Scope & Threat Model

In computer vision foundation models and multimodal systems, specific security considerations include:

- **Adversarial Input Exploits**: Specially crafted image payloads causing memory exhaustion or denial-of-service in image decoding libraries (`Pillow`, `libjpeg-turbo`).
- **Prompt Injection & Alignment Degradation**: Manipulated text prompts designed to bypass safety filters or degrade zero-shot classification integrity.
- **Model Checkpoint Integrity**: Loading untrusted weights or pickled model checkpoints. We recommend `safetensors` format and official Hugging Face / PyTorch Hub model repositories.
- **Supply Chain Vulnerabilities**: Outdated dependencies with known CVEs in upstream transformers or tensor runtime libraries.

---

## Reporting a Vulnerability

If you identify a security issue or vulnerability, please report it responsibly:

### How to Report

1. **GitHub Security Advisory (Preferred)**:
   - Go to the repository's **Security** tab.
   - Click **Report a vulnerability** to open a private advisory draft.
2. **Direct Maintainer Contact**:
   - Contact the maintainer via GitHub: [@gbhanuvigneshnaidu29052002-droid](https://github.com/gbhanuvigneshnaidu29052002-droid).

### What to Include

- A clear explanation of the vulnerability and attack vector.
- A minimal reproducible example or proof-of-concept (PoC).
- An assessment of the potential impact on inference systems or pipelines.

### Response Commitment

- Initial acknowledgment within 48 hours.
- Prompt triage, status updates, and coordinated disclosure with full attribution upon patch release (unless anonymity is preferred).

Please do not disclose potential vulnerabilities publicly through open issues or discussions.
