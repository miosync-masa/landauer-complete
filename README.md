# Landauer Complete

**Complete Thermodynamics of Information Operations: Beyond Landauer's Erasure Principle**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains simulation code for the paper:

> **Complete Thermodynamics of Information Operations: Beyond Landauer's Erasure Principle**  
> Masamichi Iizumi (Miosync, Inc.)  
> Physical Review Letters (submitted)

Landauer's principle (1961) established that erasing one bit costs at least $k_B T \ln 2$. But information systems perform four operations: **Create, Read, Update, Delete (CRUD)**. For 60 years, only Delete had known thermodynamic bounds.

**This work completes the thermodynamics of information.**

## Key Findings

| Operation | Work | Interpretation |
|-----------|------|----------------|
| **Create** | $W < 0$ (−1.5 × Landauer) | **Exothermic!** Energy released |
| **Delete** | $W > 0$ (+2.2 × Landauer) | Landauer confirmed |
| **Update** | $W \approx 0.2$ × Landauer | Partial energy recovery |
| **Maintain** | Two phases | See below |

### Maintain Phase Dichotomy

The most significant discovery: **Maintain exhibits two thermodynamic phases**.

```
Phase I (Event-Driven):   B >> k_B T
  E_maintain = Γ(T) × E_correction
  where Γ(T) ∝ exp(-B / k_B T) → 0 as T → 0
  Example: Flash memory, HDD, SSD

Phase II (Flux-Driven):   B ~ k_B T
  P_maintain = p × N (constant power)
  Example: DRAM, SRAM, biological memory

Critical Point: B_c ≈ k_B T × ln(N)
```

This is **not implementation-dependent**—it reflects fundamental thermodynamic structure.

## Installation

```bash
# Clone repository
git clone https://github.com/miosync-masa/landauer-complete.git
cd landauer-complete

# Install dependencies
pip install jax jaxlib matplotlib numpy
```

### Requirements

- Python 3.8+
- JAX (with GPU support recommended)
- Matplotlib
- NumPy

## Usage

### Run CRUD Simulation

```bash
python landauer_simulator.py
```

Output:
- `landauer_simulation_results.png` — CRUD operation costs
- Console output with detailed statistics

### Run Maintain Phase Analysis

```bash
python landauer_maintain_phases.py
```

Output:
- `landauer_maintain_phases.png` — Phase diagram and scaling relations
- Verification of the Maintain Cost Dichotomy Theorem

## File Structure

```
landauer-complete/
├── README.md
├── LICENSE
├── landauer_simulator.py        # Main CRUD simulation
├── landauer_maintain_phases.py  # Maintain phase dichotomy proof
├── paper/
│   ├── main.tex                 # PRL manuscript
│   ├── references.bib           # Bibliography
│   └── figures/                 # Paper figures
└── results/
    ├── landauer_simulation_results.png
    └── landauer_maintain_phases.png
```

## Physical Model

Single bit modeled as Brownian particle in double-well potential:

$$V(x; b, c) = x^4 - bx^2 - cx$$

Overdamped Langevin dynamics:

$$\gamma \frac{dx}{dt} = -\frac{\partial V}{\partial x} + \sqrt{2\gamma k_B T}\, \xi(t)$$

## Citation

If you use this code, please cite:

```bibtex
@article{Iizumi2025Landauer,
  author = {Iizumi, Masamichi},
  title = {Complete Thermodynamics of Information Operations: 
           Beyond Landauer's Erasure Principle},
  journal = {Physical Review Letters},
  year = {2025},
  note = {Submitted}
}
```

## Related Work

- [The Consonance Tensor](https://doi.org/10.5281/zenodo.17920244) — Mathematical framework for music analysis
- [HELA](https://github.com/miosync-masa/HELA) — Harmonic Energy Landscape Analyzer

## License

MIT License

## Author

**Masamichi Iizumi**  
Miosync, Inc.  
Email: m.iizumi@miosync.email

---

*"Information is physical. Now we know exactly how physical."*
