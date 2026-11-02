# Wired resilience: Adversity shapes genetic and mental health signature in the brain

This is the repositiory for: (preprint to come)

For any questions regarding the use of this repository, please get in touch at alsm3@cam.ac.uk

## Requirements

Python 3.11. Install the main dependencies with:

```bash
pip install numpy pandas scipy scikit-learn statsmodels pingouin matplotlib seaborn pycirclize joblib tqdm tqdm-joblib bctpy
```

## Data

ABCD data are not included in this repository: all  data is accessible on the NBDC Data Hub (https://www.nbdc-datahub.org/). This repository includes randomized data (/example_data) that can be used to run all scripts.

## Code

All functions are in `A_functions.py` and paths are set in `A_config.py`.

The scripts are alphabetized based on the order that they appear in the manuscript:

- B_sample_diversity: Basic relationships between adversity, mental health, and polygenic scores
- C_connectome_descriptives: Visualization of network properties
- D_run_PLSes: Run 3 seperate PLS models between domains (Y) and edges (X)
- E_overlap_analysis: Explore overlapping edges between PLSes and define risk- and health-aligned edges
- F_mediation_and_sliding_scale: Explore edge score mediation and sliding scale across adversity
- G_heterogenous_profile_mathced: Define 4 matched groups of to explore risk- and health-aligned edge differences between resilient and vulnerable children

## Data definitions

- **Risk-aligned:** edges positively related to adversity and/or mental health problems and negative for genetics. These are coded as 'positive' or 'pos'.
- **Health-aligned:** edges negatively related to adversity and/or mental health problems and negative for genetics. These are coded as 'negative' or 'neg'.
- **Edge groups:** edges that were stable in two PLSes. Groups are: M&A (Mental Health & Adversity), M&G (Mental Health & Genetics) and A&G (Adversity & Genetics).
- **Edge scores:** average weight of edges in a participants connectome within a specific edge group.


---

Note: Claude was used to help clean up this repository and create example data. All scripts were re-run
with original data to ensure all outputs were accurate, but this is my first attempt to use AI for
code publication, so please do let me know if something catches your eye!
