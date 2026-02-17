# Assessing Mental Health Therapeutic Capacity in AI Agents

Official repository for the paper "Assessing Mental Health Therapeutic Capacity in AI Agents", submitted to AIME 2026.
This repository contains the experimental results, analysis code, and outputs presented in the paper.

## Repository Structure

```
.
├── experimental_results_analysis.ipynb   # Main analysis notebook
├── data/
│   ├── *.json                            # Experimental results per dataset
│   └── all_results.csv                   # Aggregated results
├── output/                               # Analysis outputs
├── requirements.txt                      # Python dependencies
└── README.md
```

## Reproducing the Analysis

### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab

### Setup

```bash
# Download and extract the repository, then navigate to it
cd AIME2026-63B3

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
jupyter notebook experimental_results_analysis.ipynb
```

### Data Description

- **`data/*.json`** — Raw experimental results for each dataset evaluated in the study.
- **`data/all_results.csv`** — Aggregated dataset used for performance comparisons and statistical analyses in the notebook.


## License

This project is licensed under the [MIT License](LICENSE).