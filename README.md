# HentschManifoldFigures

This repository contains Python code and figure assets supporting the paper:

**"Null Structure from Cyclic Constraints in $\mathbb{C}^3$: A Minimalist Model of Directional Geometry from Algebraic Coupling"**  
by Patrick Hentsch

If you use this material in your own work, please cite:

Hentsch, P. (2025). Null Structure from Cyclic Constraints in $\mathbb{C}^3$: A Minimalist Model of Directional Geometry from Algebraic Coupling. Preprint. https://doi.org/10.5281/zenodo.15333416

## Overview

The figures in the manuscript were generated using custom Python scripts built with:

- `matplotlib` for visualization
- `numpy` for numerical computation

These visualizations depict the null cone geometry, screen quotient structure, and cyclic coupling of complex coordinates as described in the paper.

## Contents

- `/figures/` — Exported `.png` images used in the manuscript
- `/scripts/` — Python scripts used to generate each figure
- `requirements.txt` — Python package dependencies
- `LICENSE` — MIT License for code (see below)
- `README.md` — This file

## Usage

To generate the figures locally:

```bash
git clone https://github.com/phentsch/HentschManifoldFigures.git
cd HentschManifoldFigures
pip install -r requirements.txt
python scripts/generate_figures.py
