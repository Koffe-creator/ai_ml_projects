# Hauck-Donner Effect Simulation

An R simulation demonstrating the Hauck-Donner Effect (HDE) — a known pathology of the Wald test in logistic regression where the test statistic deflates as the effect size grows large, producing misleadingly large p-values in association studies.

## What it does

- Fixes the control proportion at π₀ = 0.5 (n = 50)
- Varies the case proportion π₁ from near 0 to near 1 across 49 iterations
- At each step, fits a logistic regression and extracts:
  - **Wald test** p-value
  - **Likelihood Ratio Test (LRT)** p-value
  - **Permutation test** p-value (20 permutations)
- Plots −log₁₀(p) vs. difference in proportions for both Wald and LRT

## Key finding

The Wald test p-value peaks and then *decreases* as the effect size grows extreme — this is the HDE. The LRT does not suffer from this pathology and is the recommended alternative.

## Output

`figures/logistic_regression_HDE.png` — comparison plot of Wald vs. LRT p-values across the full range of effect sizes.

## Usage

```r
source("logistic_hde_simulation.R")
```

No external packages required — uses base R only.
