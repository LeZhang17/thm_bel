# BEL Package

This repository provides core code structure and data required to reproduce the experiments for two Bayesian Ensemble Learning (BEL) approaches described in our manuscript named "A Bayesian Evidential Learning Framework for Safety and Performance Prediction in Thermo-Hydro-Mechanical Coupled Deep Mine Geothermal Systems":

1. **BEL–MDN**: BEL framework where the conditional distribution is modeled via a Mixture Density Network (MDN) implemented in TensorFlow Probability.  
2. **BEL–CCA**: BEL framework using classical Canonical Correlation Analysis (CCA) plus Kernel Density Estimation (KDE) via the `skbel` library. For details on the CCA implementation, see https://skbel.readthedocs.io/en/latest/.

## Data Description

All data files are provided in `data/`:

- **data_wt.pkl**  
  Production‐well temperature time‐series.  
- **data_dt.pkl**  
  Measured/ambient temperature time‐series.  
- **data_ds.pkl**  
  Target variable: drift‐stress–induced variation of friction angle (prediction horizon).
