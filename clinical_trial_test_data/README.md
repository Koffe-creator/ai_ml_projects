# Clinical Trial Analysis

Statistical analysis of clinical trial biomarker data across treatment groups and time points, using mixed-effects models and pairwise comparisons.

## Dataset
`stats_R_Python_questions_2.xlsx` — longitudinal digital count measurements for multiple biomarkers across 5 time points (DAY1, DAY8, DAY15, DAY22, DAY29) and treatment groups.

## Analysis

- **Descriptive statistics** — mean and SD by marker, time point, and treatment group
- **Visualization** — mean ± SD trajectories over time, faceted by marker
- **DAY1 vs DAY8 comparison** — paired or unpaired t-test per marker/treatment depending on subject overlap
- **Mixed-effects model** — `lmer(digital_count ~ sample_time + trt_group + (1 | sbj_id))` per marker
- **Multiple testing correction** — Bonferroni adjustment on pairwise time-point contrasts
- **Residual diagnostics** — residuals vs. fitted plots to assess homoscedasticity

## Requirements

```r
install.packages(c("readxl", "tidyverse", "ggplot2", "lme4",
                   "lmerTest", "broom", "broom.mixed", "emmeans"))
```

## Usage

Open `clinical_trial_analysis.R` in RStudio and run. The script auto-detects its location — no path changes needed.
