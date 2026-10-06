## Python Fundamentals Workbook (RA Onboarding)

This folder is a self-contained tutorial for learning Python and data-analysis
basics, built around the real data and code from **Study 3** (`../Study 3 - AAAI 2026/`):

[Moral Change or Noise? On Problems of Aligning AI With Temporally Unstable Human Feedback](https://arxiv.org/abs/2511.10032) <br>
Vijay Keswani, Cyrus Cousins, Breanna Nguyen, Vincent Conitzer, Hoda Heidari, Jana Schaich Borg, Walter Sinnott-Armstrong<br>
*AAAI Alignment Track*, 2026

Rather than generic exercises, the workbook walks through the same concepts
(dictionaries, loops, functions, pandas filtering/groupby, plotting, a simple
classifier) using the kidney-allocation pairwise-comparison data and
simplified versions of the real analysis functions in `helpers.py` and
`stability_utils.py`. Since Study 3 is finished, this is a safe practice
sandbox — nothing you run here affects the published results.

### Contents

- `Tutoral-1.ipynb` — the workbook. Read each section,
  run the example cell(s), then try the **Your turn** exercise before
  checking the solution underneath it.

### Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook "Tutoral-1.ipynb"
```

The notebook loads data directly from
`../Study 3 - AAAI 2026/data/main_response_data/main_response_data_qualtrics_participants.csv`
(no need to copy or re-download anything).

### After finishing

Once the fundamentals feel comfortable, the natural next step is
`../Study 3 - AAAI 2026/main_analysis.ipynb`, the notebook that generated every
result and figure in the paper — the workbook ends with a pointer to where
to pick that up.
