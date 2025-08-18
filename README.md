# ifct-enterprise

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Stars](https://img.shields.io/github/stars/MiguelAngelFrancoLeon/ifct-enterprise?style=social)](https://github.com/MiguelAngelFrancoLeon/ifct-enterprise)

## Overview

**ifct-enterprise** is an enterprise-grade Python framework implementing Infinite Fractal Cubes Theory (IFCT) v3.1 Geométrico-Helicoidal. It features multidimensional fractional calculus, adaptive δ_G optimization, hybrid quantum-classical simulations via Qiskit, and GPU acceleration with CuPy. Designed for production environments, it supports real-time performance monitoring, benchmarking, and scalable architectures for applications in AI, quantum computing, biological modeling, and HPC.

This library bridges theoretical fractal geometries (cubo-cilindro-helicoidal) with practical tools, minimizing energy functionals for optimal scaling. For the theoretical foundation, see [docs/v3teoria.txt](docs/v3teoria.txt).

## Features

- **Advanced Fractional Calculus**: Multidimensional derivatives using Grünwald-Letnikov, Fourier, and Caputo methods, optimized for CPU/GPU.
- **Adaptive δ_G Manager**: Real-time optimization with gradient, evolutionary, ML (RandomForest), and hybrid algorithms, including stability and data feature analysis.
- **Quantum Integration**: Qiskit-based hybrid simulations with fractal circuits and coherence preservation.
- **Performance Tools**: Parallel processing, LRU caching, memory management, and profiling decorators.
- **Enterprise Validation**: Quality scoring, error handling, logging, and auto-generated documentation.
- **Benchmarking Suite**: Scalability tests, GPU vs CPU comparisons, and resource analysis.
- **Configurable Setup**: Flexible for HPC, cloud, edge, or development via `IFCTConfig`.

## Installation

```
# Clone
git clone https://github.com/MiguelAngelFrancoLeon/ifct-enterprise.git
cd ifct-enterprise

# Environment with conda (recommended)
conda env create -f environment.yml
conda activate ifct-enterprise

# Or with pip (CPU minimal)
pip install -r requirements.txt
pip install -e .
```

Colab (CPU/GPU optional):

```
# CPU minimal
!pip -q install numpy scipy matplotlib numba joblib scikit-learn
# GPU optional (if Colab has CUDA)
# !pip -q install cupy-cuda12x
# Qiskit optional
# !pip -q install qiskit
```

## Usage

### Quickstart (CPU minimal)

```
from ifct_enterprise import IFCTConfig, EnterpriseIFCTSystem
import numpy as np

config = IFCTConfig(
    delta_G=0.921,
    use_gpu=False,          # True if Cupy available
    adaptive_delta_g=True,
    quantum_integration=False  # True if Qiskit available
)

system = EnterpriseIFCTSystem(config)
data = np.random.randn(256, 256)

res = system.full_analysis_enterprise(
    data,
    analysis_type="comprehensive",
    target_metrics={"compression_ratio": 4.0, "accuracy": 0.9}
)

print("Optimal δ_G:", res["delta_g_optimization"]["optimal_delta_g"])
print("Quality score:", res["quality_validation"]["overall_quality_score"])
```

### Quickstart (GPU + Qiskit if available)

```
try:
    import cupy as cp; HAS_GPU=True
except Exception:
    HAS_GPU=False

try:
    import qiskit; HAS_Q=True
except Exception:
    HAS_Q=False

config = IFCTConfig(use_gpu=HAS_GPU, quantum_integration=HAS_Q)
```

More examples in [examples/](examples/).

## Scope & Claims

IFCT v3.1 is a fractional calculus and simulation framework with modules for δ_G optimization, GPU support, and quantum bridging.  
Theoretical hypotheses linked to IFCT are documented in v3teoria.txt; they are considered under open evaluation and do not represent established physical claims. The software is independent and useful for fractional PDEs, stochastic modeling, and hybrid quantum-classical prototypes.

## Benchmarks (Reproducible)

```
python -m benchmarks.run_scaling --sizes 256 512 1024 --iters 3 --seed 42
python -m benchmarks.run_gpu_vs_cpu --sizes 512 1024 --iters 3 --seed 42
```

| Test             | Size   | CPU (s) | GPU (s) | Speedup  |
|------------------|--------|---------|---------|----------|
| FracFourier3D    | 512³   | 12.3    | 5.7     | 2.16×    |
| FracFourier3D    | 1024³  | 98.1    | 42.8    | 2.29×    |

Hardware: RTX 3090, CUDA 12.2, i9-12900K, Linux 6.8, Python 3.11, CuPy 13.0, NumPy 2.0.

## Documentation

- [API Docs](docs/ifct_enterprise_api_docs.md)
- [Config Guide](docs/ifct_enterprise_config_guide.md)
- [Theory](docs/v3teoria.txt)

## Contributing

Fork, branch, commit, and PR! Focus on math enhancements, new integrations, or benchmarks.

## Citing

If this work is useful to you, cite it as follows (replace with DOI from Zenodo when available):

```
@software{FrancoLeon_IFCT_Enterprise_2025,
  author  = {Franco León, Miguel Ángel},
  title   = {IFCT v3.1 Enterprise Grade},
  year    = {2025},
  url     = {https://github.com/MiguelAngelFrancoLeon/ifct-enterprise},
  version = {v3.1},
  license = {MIT}
}
```

## License

MIT License - see [LICENSE](LICENSE).

## Acknowledgments

Built with insights from fractal theories and IBM reviews. Star if useful! 🚀
