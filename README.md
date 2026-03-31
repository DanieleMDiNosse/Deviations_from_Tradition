# Deviations from Tradition: Stylized Facts in the Era of DeFi

This repository contains the research code associated with the paper:

> Di Nosse, D. M., Gatta, F., Lillo, F., & Jaimungal, S. (2025). *Deviations from Tradition: Stylized Facts in the Era of DeFi*. arXiv preprint arXiv:2510.22834.

## Overview

Decentralized Exchanges (DEXs) have become a major component of modern financial markets, yet their microstructure differs fundamentally from that of traditional exchanges. Instead of limit order books, most DEXs rely on automated market makers, and Uniswap v3 adds a further structural innovation through concentrated liquidity.

The goal of the project is to study how these design choices reshape the classical stylized facts of market microstructure. Using high-frequency, event-level data from the 24 most active Uniswap v3 pools during 2023 and 2024, the paper documents how prices, liquidity, order flow, and event transitions differ from the benchmark patterns commonly observed in traditional finance.

A central message of the paper is that these deviations are not random noise. They are tightly connected to the specific organization of on-chain markets: block-based execution, concentrated liquidity, and the activity of Maximal Extractable Value (MEV) searchers, including sandwich attackers and Just-in-Time liquidity providers. The resulting empirical patterns are both economically relevant and useful for the construction of realistic high-frequency simulators of DEX dynamics.

## What This Repository Contains

The repository is organized around a single core module plus example notebooks.

### Core module

- [utils_DevTrad.py](./utils_DevTrad.py): main user-facing module.
- [utils_DevTrad_Rust](./utils_DevTrad_Rust): Rust crate compiled and imported from Python for selected performance-critical routines.
- [utils_DevTrad_Python.py](./utils_DevTrad_Python.py): Python reference implementation used during development and validation.

The intended entry point for users is `utils_DevTrad.py`. The Rust component is an implementation detail for acceleration: the public API remains Python-based.

### Example notebooks

- [Single_Pool_Analysis_Prototype.ipynb](./Single_Pool_Analysis_Prototype.ipynb): single-pool workflow and illustrative analysis.
- [Multiple_Analysis.ipynb](./Multiple_Analysis.ipynb): multi-pool analysis and broader empirical workflow.

These notebooks provide examples of how the functions in `utils_DevTrad` can be combined to reproduce parts of the analysis developed in the paper.

## Main Analytical Components

The `utils_DevTrad` module includes functions for several classes of tasks:

### MEV and event-structure analysis

- `find_jit`: detect Just-in-Time liquidity episodes.
- `find_sandwich`: detect sandwich attacks and related event roles.
- `find_echo`: identify echo-swap patterns.
- `mix_backrun_volumes`: measure mixed backrun volumes.
- `liq_change`: track liquidity changes and event-time liquidity structure.

### Time-series and microstructure statistics

- `transition_probabilities`: transition probabilities across event types.
- `distribution_daily`: intraday distributions and confidence intervals.
- `bootstrap_iid_autocorr`: IID bootstrap for autocorrelation analysis.
- `long_memory_test`: long-memory diagnostics and Hurst-style metrics.
- `long_memory_H_dist`: bootstrap/distributional analysis of long-memory estimators.
- `provision_summary`: summary statistics for liquidity provision behavior.

### Visualization and supporting analysis

- `plot_acf`, `plot_acf_long_memory`
- `periodic_analysis`
- `rescale_data`

In practice, the module is designed for researchers working with event-level Uniswap v3 data who want tools for MEV detection, return/liquidity diagnostics, long-memory analysis, and transition-structure estimation.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Stylized_Facts
```

### 2. Create the conda environment

An environment file is provided:

- [environment.yml](./environment.yml)

Create and activate the environment with:

```bash
conda env create -f environment.yml
conda activate DevTrad
```

### 3. Compile the Rust extension

Part of the library is implemented in Rust and exposed to Python via `PyO3` and `maturin`.

```bash
cd utils_DevTrad_Rust
maturin develop --release
cd ..
```

After this step, the main module can be imported from Python:

```python
import utils_DevTrad
```

## Minimal Usage

A typical workflow is:

```python
import utils_DevTrad as udt

# Example: MEV / event labeling
sandwich_labels = udt.find_sandwich(block_number, wallet, event)

# Example: long-memory diagnostics
results = udt.long_memory_test(series)

# Example: transition probabilities
probs = udt.transition_probabilities(event, shifts=[1, 2, 3])
```

The notebooks included in the repository show fuller research-oriented workflows, from event cleaning and labeling to stylized-fact estimation and figure generation.

## Notes on Reproducibility

This repository is research code associated with the paper rather than a polished production package. The goal is transparency and reproducibility of the empirical analysis.

For reproducible use, it is recommended to:

1. Create the environment from `environment.yml`.
2. Compile the Rust crate locally with `maturin develop --release`.
3. Use the notebooks as reference workflows.
4. Keep the Python wrapper `utils_DevTrad.py` as the public interface.

## Why This Paper May Be of Interest

The paper speaks to several communities at once:

- market microstructure researchers interested in how DeFi differs from centralized markets,
- quantitative researchers studying high-frequency crypto data,
- researchers working on MEV, block-building, and on-chain execution,
- practitioners interested in realistic simulation of DEX activity.

Its main contribution is to show that DEXs are not simply another venue where traditional stylized facts reappear unchanged. Instead, the combination of AMM design, concentrated liquidity, blockchain execution, and MEV creates a distinct microstructure with its own empirical regularities.

## Citation

If you use this repository, please cite the associated paper:

Di Nosse, D. M., Gatta, F., Lillo, F., & Jaimungal, S. (2025). *Deviations from Tradition: Stylized Facts in the Era of DeFi*. arXiv preprint arXiv:2510.22834.
