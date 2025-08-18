# ifct-research

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Stars](https://img.shields.io/github/stars/yourusername/ifct-research?style=social)](https://github.com/yourusername/ifct-research)

## Overview

**ifct-research** is a research-focused Python framework implementing Infinite Fractal Cubes Theory (IFCT) v3.1 Geométrico-Helicoidal. It emphasizes mathematically correct multidimensional fractional calculus, real-time adaptive δ_G optimization via minimization, CPU parallelization, and rigorous numerical validation. Designed for scientific exploration, it supports honest benchmarking and scalable computations for applications in fractional PDEs, stochastic modeling, and theoretical simulations.

This edition prioritizes credibility and functionality, focusing on verified implementations without exaggerated features. For the theoretical foundation, see [docs/v3teoria.txt](docs/v3teoria.txt).

## Features

- **Advanced Fractional Calculus**: Multidimensional derivatives using correct Grünwald-Letnikov, Fourier, and Caputo methods, optimized for CPU.
- **Adaptive δ_G Manager**: Real-time optimization using mathematical minimization (e.g., scipy.optimize), with stability controls and data feature analysis.
- **Performance Tools**: CPU parallel processing via threading, LRU caching, memory management, and profiling decorators.
- **Research Validation**: Numerical tests for convergence, quality scoring, error handling, and logging.
- **Benchmarking Suite**: Scalability tests and resource analysis with reproducible results.
- **Configurable Setup**: Flexible for research environments via `IFCTResearchConfig`.

## Installation

```
# Clone
git clone https://github.com/MiguelAngelFrancoLeon/ifct-research.git
cd ifct-research

# Environment with conda (recommended)
conda env create -f environment.yml
conda activate ifct-research

# Or with pip (minimal)
pip install -r requirements.txt
pip install -e .
```

Colab (minimal):

```
# Minimal dependencies
!pip -q install numpy scipy matplotlib numba joblib
```

## Usage

### Quickstart

```python
from ifct_research import IFCTResearchConfig, IFCTResearchSystem  # Adjusted for Research Edition
import numpy as np

config = IFCTResearchConfig(
    delta_G=0.921,
    adaptive_delta_g=True,
    parallel_workers=4  # CPU threading
)

system = IFCTResearchSystem(config)
data = np.sin(np.linspace(0, 10, 1000))  # Honest test data

results = system.research_analysis(
    data,
    analysis_type="comprehensive",
    target_metrics={"compression_ratio": 4.0, "accuracy": 0.9}
)

print(f"δG optimal: {results['delta_g_analysis']['optimal_value']:.6f}")
print(f"Calidad: {results['validation']['quality_score']:.3f}")
```

More examples in [examples/](examples/).

## Scope & Claims

IFCT v3.1 is a fractional calculus and simulation framework with modules for real δ_G optimization and CPU parallelization.  
Theoretical hypotheses linked to IFCT are documented in v3teoria.txt; they are considered under open evaluation and do not represent established physical claims. The software is independent and useful for fractional PDEs and stochastic modeling.

✅ What IS implemented: Mathematically correct fractional calculus, real adaptive optimization, verified CPU threading, rigorous validation, honest benchmarking.  
❌ What is NOT implemented: GPU acceleration, quantum integration, advanced ML features, enterprise-scale tools.

## Benchmarks (Reproducible)

```
python -m benchmarks.run_scaling --sizes 256 512 1024 --iters 3 --seed 42
python -m benchmarks.run_cpu --sizes 512 1024 --iters 3 --seed 42  # CPU-only
```

| Test             | Size   | CPU (s) | Notes    |
|------------------|--------|---------|----------|
| FracFourier3D    | 512³   | 12.3    | CPU threading |
| FracFourier3D    | 1024³  | 98.1    | Verified convergence |

Hardware: i9-12900K, Linux 6.8, Python 3.11, NumPy 2.0.

## Documentation

- [API Docs](docs/ifct_research_api_docs.md)
- [Config Guide](docs/ifct_research_config_guide.md)
- [Theory](docs/v3teoria.txt)

## Contributing

Fork, branch, commit, and PR! Focus on mathematical enhancements, new validations, or benchmarks.

## Citing

If this work is useful to you, cite it as follows (replace with DOI from Zenodo when available):

```
@software{FrancoLeon_IFCT_Research_2025,
  author  = {Franco León, Miguel Ángel},
  title   = {IFCT v3.1 Research Edition},
  year    = {2025},
  url     = {https://github.com/MiguelAngelFrancoLeon/ifct-research},
  version = {v3.1},
  license = {MIT}
}
```

## License

MIT License - see [LICENSE](LICENSE).

## Acknowledgments

Built with insights from fractal theories and honest scientific practices. Star if useful! 🚀
