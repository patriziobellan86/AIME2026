# Assessing Mental Health Therapeutic Capacity in AI Agents

Official repository for the paper **"Assessing Mental Health Therapeutic Capacity in AI Agents"**, submitted to **AIME 2026**.  
This repository contains the experimental results, analysis code, and outputs presented in the paper.

## Paper Abstract

Large Language Models (LLMs) have been widely investigated for conversational support in mental health.  
However, their therapeutic reliability in empathy and perspective-taking remains uncertain.  
We aim to fill this gap by introducing a comprehensive evaluation of the Pool of Experts framework, a multi-agent approach that enables role-specific identities without retraining the underlying model.  
Beyond its effectiveness, this framework provides a controlled test-bed to study whether personality framing induces measurable behavioral variation across roles and tasks.  
We systematically assess the Pool of Experts capability in empathy and theory-of-mind-oriented benchmarks and compare structured multi-agent orchestration with less structured conditions.  
Results demonstrate three main findings.  
First, architectural orchestration with deliberative aggregation consistently improves performance: a Final Decision Maker agent improves accuracy by up to 3.5 percentage points compared with individual experts.  
This aggregation mechanism also provides superior error recovery compared to majority voting.  
Second, such improvements generalize robustly across model families without requiring larger parameter scales.  
Compact and large-scale architectures achieve comparable benefits.  
Third, process-oriented frameworks yield the strongest gains, while theory-of-mind tasks benefit more reliably from structured orchestration than empathy tasks.  
This suggests perspective-taking is currently more tractable for optimization than affective alignment.  
These findings support structured multi-agent orchestration as a reliable way to improve socio-cognitive behaviors in mental-health-oriented LLM systems.

**Keywords:** Agent Personality Framing; Mental Health; Empathy; Theory of Mind; Multi-Agent Systems

## Repository Structure

```
.
├── experimental_results_analysis.ipynb   # Main analysis notebook
├── experiment-luncher.py                 # Interactive experiment runner
├── fix-aptths.py                         # Utility to fix broken absolute paths in .json/.py
├── data/
│   ├── *.json                            # Experimental results per dataset
│   └── all_results.csv                   # Aggregated results
├── output/                               # Analysis outputs
├── requirements.txt                      # Python dependencies (analysis environment)
└── README.md
```

## Reproducing the Analysis

### Prerequisites

- Python 3.13.5+
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

## Re-running the Experiments

The experiments can be re-executed via the interactive launcher script:

### 1) Run `experiment-luncher.py`

```bash
python experiment-luncher.py --project-root /absolute/path/to/Experiments
```

This script provides an interactive flow to:
- select the `dataset_type`
- select the dataset folder (supports nested structures)
- optionally filter configs (baseline vs PoE)
- run either selected configs or all configs in a folder
- choose which pipeline stages to execute:
  - `initialization`
  - `experiments`
  - `final_decision_maker`

### 2) Fix broken paths (if needed) with `fix-paths.py`

If your config files or scripts contain machine-specific absolute paths (e.g., `/Users/.../ExperimentsVirtualPsy/...`) and the experiments fail because paths cannot be resolved, use:

```bash
python fix-aptths.py
```

Before running it, open `fix-paths.py` and set the variables at the top of the file:
- `ROOT_DIR` (the folder to scan recursively)
- `OLD` (the broken/old path prefix)
- `NEW` (the new path prefix to apply)

The utility will replace occurrences in both **`.json` and `.py`** files across folders and subfolders (with optional dry-run/backup depending on how the script is configured).

## License

This project is licensed under the [MIT License](LICENSE).
