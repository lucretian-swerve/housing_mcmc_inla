# MCMC vs INLA for Housing Price Modeling

This repository contains code and documentation for a project comparing two Bayesian inference techniques—Markov Chain Monte Carlo (MCMC) and Integrated Nested Laplace Approximation (INLA)—applied to a housing price dataset.

## Project Overview

We model log-transformed housing prices based on predictors such as:
- Number of bedrooms and bathrooms
- Square footage of living space
- Square footage above ground
- Building grade

The project includes:
- Custom Metropolis-Hastings implementation for Bayesian linear regression
- Equivalent INLA model using the `R-INLA` package
- Performance comparison based on RMSE, MAE, and $R^2$
- Posterior summaries and credible intervals

## Contents

- `mcmc_model.R`: MCMC sampling code
- `inla_model.R`: INLA model specification and execution
- `housing_data.csv`: Cleaned dataset (if allowed)
- `report.tex`: Final report in LaTeX format
- `figures/`: Plot outputs for report

## Reproducibility

All code is written in R and can be run with:
```r
source("mcmc_model.R")
source("inla_model.R")

