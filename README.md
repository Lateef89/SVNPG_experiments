# SVNPG Experiments

Reproducible code for the paper:
**"Stochastic Proximal Gradient Methods with Finite-Time Support Identification for Non-Smooth Composite Optimization"**

## Structure

- `svnpg_core.py` — Core algorithms: SVNPG, ProxSGD, ProxSVRG-L1, Stochastic PALM, MCP baseline, plus LAD/Logistic losses and Capped-ℓ1/MCP penalties.
- `experiments.py` — Experiment runners for synthetic LAD, large-scale dense LAD, RCV1, and News20.
- `generate_figures.py` — Plotting scripts to reproduce all figures and tables in the manuscript.
- `requirements.txt` — Python dependencies.

## Quick start

```bash
pip install -r requirements.txt
python experiments.py          # Run synthetic experiments
python generate_figures.py     # Generate all figures
```

## Reproducibility

All experiments report the median over 20 independent runs with interquartile bands (IQR). Random seeds are fixed per run. The conda-lock environment will be archived upon acceptance.

## Contact

Corresponding author: Yekini Shehu (yekini.shehu@zjnu.edu.cn)
