# From Erasure to CRUD: A Unified Information-Thermodynamic Principle

**Landauer's principle is not just about deletion.**

This repository contains the complete simulation code, data, and manuscript source for our work demonstrating that Landauer's erasure principle is a special case of a broader CRUD (Create, Read, Update, Delete) information thermodynamics.

## Key Result

All four CRUD operations are governed by a single nonequilibrium free-energy bound:

$$\langle W \rangle \geq \Delta F_{\mathrm{eq}} + kT \, \Delta D_{\mathrm{KL}}^{\mathrm{full}} \equiv \Delta F_{\mathrm{total}}$$

| Operation | ΔF_total | Character |
|-----------|----------|-----------|
| **Create** (blank₀ → 1) | ≈ 0 | Write-like, not kT ln 2 limited |
| **Read** (non-destructive) | 0 (by construction) | Measurement cost via correlation |
| **Update** (unknown → 1) | ≈ kT ln 2 | Erase-like component |
| **Delete** (unknown → blank₀) | ≈ kT ln 2 | Standard Landauer case |

Delete is not special because it is the only thermodynamic operation — it is special because it maximizes macro KL compression. The CRUD framework unifies all four operations under one accounting scheme.

## Repository Structure

```
.
├── README.md
├── LICENSE
├── crud_simulator.py            # Single-file JAX simulator (GPU)
├── figures/
│   ├── fig1_crud_overview.pdf   # Integrated 6-panel CRUD figure
│   └── fig2_read_scatter.pdf    # System-meter scatter for Read
├── data/
│   ├── *_core_summary_*.csv     # Core CRUD results
│   └── *_relax_scan_*.csv       # Quasi-static relaxation scan
└── paper/
    ├── From_Erasure_to_CRUD.pdf # Manuscript (elsarticle, PLA format)
```

## Simulation Features

- **Unified codebase**: Create, Read, Update, Delete from a single script
- **Full thermodynamic accounting**: ΔF_eq + kT ΔD_KL decomposed into macro (logical) and intra-basin (nonequilibrium distortion) contributions
- **Read as measurement**: System-meter coupling dynamics with mutual information estimation and Sagawa–Ueda bound verification
- **Write calibration**: Analytic solve + endpoint micro-correction to maintain ε_write ≈ 0.01
- **Quasi-static scan**: Relaxation sweep (1, 2, 5, 10, 20 steps) separating protocol dissipation from state-function contributions
- **JAX on GPU**: Vectorized Euler–Maruyama integration with JIT compilation

## Quick Start

```bash
# Requirements: JAX (with GPU support recommended), NumPy, Matplotlib
pip install jax jaxlib numpy matplotlib

# Run the full simulation
python crud_simulator.py
```

This produces:
- CSV data tables for core results and relaxation scan
- `fig1_crud_overview.pdf` — integrated 6-panel CRUD thermodynamic figure
- `fig2_read_scatter.pdf` — system-meter correlation scatter for Read

## Model

Double-well potential memory:

$$V(x; b, c) = x^4 - bx^2 - cx$$

- $x < 0$: logical 0, $x > 0$: logical 1
- Four-stage protocol: barrier lowering → bias → barrier raising → unbias
- Overdamped Langevin dynamics at temperature $T$
- Read: coupled system-meter with ramped coupling $g(t)$

## Core Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| T | 1.0 | Temperature |
| b_hi | 4.0 | High barrier |
| b_lo | 0.2 | Low barrier |
| n_λ | 2000 | Protocol steps |
| N_particles | 10000 | Ensemble size (core) |
| ε_target | 0.01 | Write error target |
| g_max | 2.0 | Read coupling strength |

## Citation

If you use this code or build upon our CRUD framework, please cite:

```bibtex
@article{Iizumi2026crud,
  author  = {Masamichi Iizumi},
  title   = {From Erasure to CRUD: A Unified Information-Thermodynamic Principle},
  journal = {},
  year    = {2026},
  note    = {Submitted}
}
```

## Authors

**Masamichi Iizumi** — Theory, framework design, direction  
**Tamaki Iizumi** — Code architecture, numerical implementation, analysis, co-creation

Miosync, Inc., Tokyo, Japan

## Related Work

This work builds on and connects to:

- Landauer (1961) — Original erasure principle
- Bennett (1982) — Thermodynamics of computation review
- Sagawa & Ueda (2010, 2012) — Measurement thermodynamics
- Boyd, Mandal & Crutchfield (2018) — Modularity dissipation beyond Landauer
- Wolpert et al. (2024) — Stochastic thermodynamics of computation
- Manzano et al. (2024) — Absolute irreversibility in computation
- Hsieh (2025) — Dynamical Landauer principle

## License

MIT License

---

*"Landauer's principle should be understood not as a law for deletion alone, but as the deletion-specific limit of a broader CRUD information thermodynamics."*
