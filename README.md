# Deviations from Tradition: Stylized Facts in the Era of DeFi

This repository contains the research code associated with the paper:

> Di Nosse, D. M., Gatta, F., Lillo, F., & Jaimungal, S. (2025). *Deviations from Tradition: Stylized Facts in the Era of DeFi*. arXiv preprint arXiv:2510.22834.

## Overview

Decentralized Exchanges (DEXs) have become a major component of modern financial markets, yet their microstructure differs fundamentally from that of traditional exchanges. Instead of limit order books, most DEXs rely on automated market makers, and Uniswap v3 adds a further structural innovation through concentrated liquidity.

The goal of the project is to study how these design choices reshape the classical stylized facts of market microstructure. Using high-frequency, event-level data from the 24 most active Uniswap v3 pools during 2023 and 2024, the paper documents how prices, liquidity, order flow, and event transitions differ from the benchmark patterns commonly observed in traditional finance.

A central message of the paper is that these deviations are not random noise. They are tightly connected to the specific organization of on-chain markets: block-based execution, concentrated liquidity, and the activity of Maximal Extractable Value (MEV) searchers, including sandwich attackers and Just-in-Time liquidity providers. The resulting empirical patterns are both economically relevant and useful for the construction of realistic high-frequency simulators of DEX dynamics.

## What This Repository Contains

The repository is organized around a single core module and two example notebooks.

### Core module

- [utils_DevTrad.py](./utils_DevTrad.py): main user-facing module.
- [utils_DevTrad_Rust](./utils_DevTrad_Rust):(hidden) Rust crate compiled and imported from Python.

In practice, the module is designed for analyzing event-level Uniswap v3 data or, generally, data coming from Concentrated Liquidity exchanges (up to proper columns naming). The data is provided as a `pandas.DataFrame`; each row is for an event. The columns used by the module functions are:
- `amount`: amount of the liquidity minted or burnt; not considered (`numpy.nan`) if the event is a swap.
- `amount0`: if the event is a swap, it's the delta in the token X reserves of the pool (thus, it's positive if `Event=="Swap_X2Y"`); otherwise, it's the absolute amount of token X minted or burnt.
- `amount1`: if the event is a swap, it's the delta in the token Y reserves of the pool (thus, it's negative if `Event=="Swap_X2Y"`); otherwise, it's the absolute amount of token Y minted or burnt.
- `block_number`: number of the block.
- `Event`: string describing the type of event: "Swap_X2Y", "Swap_Y2X", "Mint", "Burn".
- `liquidity`: active liquidity after the swap; not considered (`numpy.nan`) if the event is a mint or a burn.
- `log_index`: position inside the block.
- `price`: price after the swap; not considered (`numpy.nan`) if the event is a mint or a burn.
- `tick_lower`: it's the upper tick of the liquidity provision range minted or burnt; not considered (`numpy.nan`) if the event is a swap.
- `tick_upper`: it's the upper tick of the liquidity provision range minted or burnt; not considered (`numpy.nan`) if the event is a swap.
- `tick`: price tick (defined as the floor of the price log base 1.0001) after the swap; not considered (`numpy.nan`) if the event is a mint or a burn.
- `wallet`: wallet sending the order.

The module contains tools for MEV detection, return/liquidity diagnostics, long-memory analysis, and transition-structure estimation.

### Example notebooks

- [Single_Pool_Analysis_Prototype.ipynb](./Single_Pool_Analysis_Prototype.ipynb): Pipeline for the analysis of a single pool at once. It contains information on the time series, quantity of MEV events, transition probabilities from one event to another, and various AutoCorrelation Functions.
- [Multiple_Analysis.ipynb](./Multiple_Analysis.ipynb): Code for the analysis of multiple pools at once, comparing statistics and properties of each of them. The pools are often divided according to the pair type or the pool fee tier.

These notebooks provide examples of usage of the module `utils_DevTrad` and its functions. It shows of the results in the paper have been obtained.

## Main Analytical Components

The functions in `utils_DevTrad` can be grouped according to their focus:

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

- `plot_acf`, `plot_acf_long_memory`: plot the ACF of a given quantity
- `periodic_analysis`
- `rescale_data`

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/DanieleMDiNosse/Deviations_from_Tradition
cd Deviations_from_Tradition
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

The notebooks included in the repository show the entire research-oriented workflows, from event cleaning and labeling to stylized fact estimation and figure generation.

## Notes on Reproducibility

This repository is research code associated with the paper rather than a polished production package. The goal is transparency and reproducibility of the empirical analysis.

For reproducible use, it is recommended to:

1. Create the environment from `environment.yml`.
2. Compile the Rust crate locally with `maturin develop --release`.
3. Use the notebooks as reference workflows.
4. Keep the Python wrapper `utils_DevTrad.py` as the public interface.

## Citation

If this repository is useful for your research, please cite our paper:

Di Nosse, D. M., Gatta, F., Lillo, F., & Jaimungal, S. (2025). *Deviations from Tradition: Stylized Facts in the Era of DeFi*. arXiv preprint arXiv:2510.22834.
