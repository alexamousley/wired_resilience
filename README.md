# Wired resilience: Adversity shapes genetic and mental health signature in the brain

This is the repositiory for: (preprint to come)

For any questions regarding the use of this repository, please get in touch at alsm3@cam.ac.uk

## Requirements

Python 3.11. Install the main dependencies with:

```bash
pip install numpy pandas scipy scikit-learn statsmodels pingouin matplotlib seaborn pycirclize joblib tqdm tqdm-joblib bctpy
```

## Data

ABCD data are not included in this repository: access is available through the [NIMH Data Archive](https://nda.nih.gov/abcd). This repository includes randomized data (in /example_data) that cane
be used to run all scripts.

## Code

All functions are in `A_functions.py` and paths are set in `A_config.py`.

The scripts are alphabetized based on the order that they appear in the manuscript:

- B_sample_diversity: Basic relationships between adversity, mental health, and polygenic scores
- C_connectome_descritives: Visualization of network properties
- D_run_PLSes
- E_overlap_analysis
- F_mediation_and_sliding_scale
- G_heterogenous_profile_mathced

## Data definitions

- **Risk-aligned:** edges positively related to adversity and/or mental health problems and negative for genetics. These are coded as 'positive' or 'pos'.
- **Health-aligned:** edges negatively related to adversity and/or mental health problems and negative for genetics. These are coded as 'negative' or 'neg'.
- **Edge groups:** edges that were stable in two PLSes. Groups are: M&A (Mental Health & Adversity), M&G (Mental Health & Genetics) and A&G (Adversity & Genetics).
- **Edge scores:** average weight of edges in a participants connectome within a specific edge group.


---

Note: Claude was used to help clean up this repository and create example data. All scripts were re-run
with original data to ensure all outputs were accurate, but give this is my first attempt to use AI for
code publication, please do let me know if something catches your eye!
