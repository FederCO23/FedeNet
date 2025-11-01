# Changelog

All notable changes to **FedeNet** will be documented in this file.  
This project follows [Semantic Versioning](https://semver.org/).

---

## [0.2.0] - 2025-11-01  
### Major Update — Modular Package, ONNX Export & Full CI  

**Highlights**  
- Refactored into a clean Python package structure (`fedenet/`) with modular files and typing.  
- Added **FedeNetTiny**, a dual-stream CNN integrating spatial and frequency maps.  
- Introduced robust preprocessing pipeline (`ResizePad`, `AppendFrequencyMaps`, `NormalizeRGBOnly`).  
- Added ONNX export (`export.py`) and `torch.hub` entrypoint (`hubconf.py`).  
- Implemented full CI workflow with linting (ruff), type checking (mypy), and tests (pytest + coverage).  
- Achieved **94% total coverage** and 100% pass rate.  

**Summary**  
This release transforms *FedeNet* into a maintainable, production-ready library with clear structure, typed modules, and reproducible ONNX export.


## [0.2.0] - 2025-11-01
### Major Update — Library Refactor & Full Test Coverage
This version transforms **FedeNet** from a single-file prototype into a clean, modular Python package with professional-grade development standards.

#### Structure & Organization
- Split the original monolithic `FedeNet.py` into a structured package:

```
fedenet/
├── init.py
├── models.py
├── transforms.py
├── weights.py
├── export.py
├── utils.py
├── hubconf.py
└── tests/
```

- Added `pyproject.toml`, `ruff.toml`, and `.mypy.ini` for type checking and linting consistency.
- Introduced proper `__version__` in `fedenet/__init__.py`.

#### Core Model
- Implemented **FedeNetTiny**, a dual-stream CNN for efficient binary classification.
- Anti-aliased stem (`StemAA`) for smoother downsampling.
- Separate **Spatial** and **Frequency** streams merged via `FedeHead`.
- Depthwise separable convolutions and SiLU activations.
- Added optional pretrained RGB weight loading for transfer learning.

#### Data & Transforms
- New modular transform utilities:
- `ResizePad`: Aspect-ratio preserving resize + pad.
- `AppendFrequencyMaps`: Generates Sobel, Laplacian, High-Pass, and Local Variance maps.
- `NormalizeRGBOnly`: ImageNet-based normalization on RGB only.
- Added high-level factories:
- `train_540x960()` and `inference_540x960()` for consistent preprocessing.

#### Export & Reproducibility
- Added `export.py` for ONNX export with dynamic batch axes.
- Ensured compatibility with PyTorch `torch.onnx.export` (mypy-clean).
- Added `hubconf.py` for `torch.hub` model loading.

#### Quality & Testing
- Achieved **100% pass rate** across:
- `pytest` — 4/4 tests passing.
- `ruff` — no linting issues.
- `mypy` — no type errors in 7 modules.
- CI pipeline (`.github/workflows/ci.yaml`) updated to include lint + type + test stages.

#### Documentation
- Added comprehensive docstrings, inline comments, and usage examples.
- Improved readability, style consistency, and type annotations.

---

## [0.1.0] - 2025-10-15
### Initial Prototype
- First working version of **FedeNet**, implemented as a single file (`FedeNet.py`).
- Included a custom CNN architecture for binary classification (flip vs notflip).
- Early implementation of frequency-map preprocessing and training pipeline.

---

### Roadmap
Planned for **v0.3.0**:
- Release minimal `examples/` training script and Google Colab demo.
- Improve ONNX export documentation and sample inference.

---

**Author:** Federico Bessi  
**License:** Apache 2.0 [LICENSE](https://github.com/FederCO23/FedeNet/blob/main/LICENSE)  
**Repository:** [https://github.com/FederCO23/FedeNet](https://github.com/FederCO23/FedeNet)