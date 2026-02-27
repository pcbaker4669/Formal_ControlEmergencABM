[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18148012.svg)](https://doi.org/10.5281/zenodo.18148012)

# An Agent-Based Disruption Generator for Classrooms (Lognormal Risk + Gamma–Poisson Counts)

Reference implementation for the disruption-only agent-based model (ABM) used in the paper:

**An Agent-Based Disruption Generator for Classrooms Using Lognormal Risk and Gamma–Poisson Counts**  
Peter C. Baker (George Mason University)

## What this model does

The model generates **incident counts** for a classroom setting with fixed rosters. Each student is an agent with a stable latent disruption propensity (“risk”) drawn from a **lognormal distribution** (between-student heterogeneity). For each class session (represented here as one **class-day record**), incidents are generated using a **Gamma–Poisson mixture**, producing **overdispersed** (bursty) totals with occasional spikes.

This repository is designed as a **reusable core generator** that can be embedded into larger classroom ABMs (instructional time loss, learning dynamics, interventions, etc.).

## Incident definition (measurement)

**Incident:** one discrete rule-/norm-violation event counted within a single observation window (a “class-period” or “class session”).  
In this implementation, the simulation time step is labeled as **day**, and the main record is **one row per class per day** (`class_day_records`). If your empirical study uses a different observation window (e.g., 45 vs 90 minutes), treat the simulated “day” as *one observation window* and calibrate rates accordingly.

Incidents are **not severity-weighted** in this implementation.

## Repository layout

- `model_core.py` — core ABM components: `Params`, `Student`, `Model`, `simulate()`  
- `analysis.py` — analysis + plotting utilities used to produce paper artifacts  
- `main.py` — orchestration script that reads `config.json`, runs replications, and saves tables/figures  
- `README.md` — this file

## Requirements

- Python 3.10+ recommended
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
```

## Quickstart

### 1) Create a `config.json`

`main.py` reads a JSON config file from the project root. Two optional keys control output folders:

- `figures_dir` (default: `"figures"`)
- `data_dir` (default: `"data"`)

All other keys are passed into `Params(...)`.

Example `config.json` (paper-style baseline):

```json
{
  "seed": 1,
  "n_students": 300,
  "class_size": 30,
  "n_days": 90,

  "risk_mu": -0.78,
  "risk_sigma": 1.6,
  "nb_k": 0.5,
  "inc_base_rate": 0.24,
  "at_risk_top_n": 3,

  "figures_dir": "figures",
  "data_dir": "data"
}
```

### 2) Run the paper artifacts

```bash
python main.py
```

By default, `main.py` runs **50 replications** (`seed=1..50`) and produces:

- **Table 1** (baseline summary metrics, mean ± SD across replications)
- **Figure 1** (Lorenz curve: concentration of incidents across students)
- **Figure 2** (CCDF: tail / spikes in class-session incident counts)
- Tail probabilities printed to console: `P(X >= 10)`, `P(X >= 20)`, `P(X >= 30)`

It also runs one-factor-at-a-time (OFAT) sensitivity sweeps (Figures 3–5).

### Output files

With the default folder names, outputs are written to:

- `data/table1_baseline.csv` (mean ± SD across seeds)
- `data/table1_baseline_per_run.csv` (per-seed summaries)
- `figures/fig1_lorenz_concentration.png` and `.pdf`
- `figures/fig2_ccdf_class_counts.png` and `.pdf`
- `figures/fig3_sweep_risk_sigma_top5.png` and `.pdf`
- `figures/fig4_sweep_nb_k_varmean.png` and `.pdf`
- `figures/fig5_sweep_inc_base_rate_level.png` and `.pdf`

## Using the generator programmatically

```python
from model_core import Params, simulate

p = Params(
    seed=1,
    n_students=300,
    class_size=30,
    n_days=90,
    risk_mu=-0.78,
    risk_sigma=1.6,
    nb_k=0.5,
    inc_base_rate=0.24,
    at_risk_top_n=3,
)

out = simulate(p)

# Core outputs
history = out["history"]                      # one row per day (overall totals)
class_day_records = out["class_day_records"]  # one row per class per day
students = out["students"]                    # per-student totals + metadata
summary = out["summary"]                      # aggregate metrics (stable keys)
```

## Model structure (core)

- **Params** (dataclass): configuration and parameters  
- **Student**: agent with stable latent `risk` and accumulated `incidents_total`  
- **Model**:
  - draws student risks (lognormal)
  - partitions students into fixed class rosters
  - simulates `n_days` time steps
  - records one count per class per day and computes summary metrics

## Parameters and interpretation

Key parameters (see `Params` in `model_core.py`):

- **`inc_base_rate`**: sets overall incident level (scales expected counts linearly)
- **`risk_sigma`**: controls concentration/inequality across students (higher values → more unequal contribution)
- **`nb_k`**: controls burstiness/overdispersion (lower values → heavier tails and more spikes)
- **`risk_mu`**: shifts overall system load via the lognormal mean (often calibrated jointly with `inc_base_rate`)
- **`at_risk_top_n`**: defines an “at-risk” subgroup within each class as the top‑N risk students (fixed for a run)

Notes:
- `model_core.py` defines defaults for all parameters; `config.json` is the intended way to set paper baselines and calibration targets.
- In `Model.step_day`, each student’s expected rate is `inc_base_rate * risk`, and counts are generated via a Gamma–Poisson mixture.

## Reproducibility

The model uses NumPy’s RNG (`numpy.random.default_rng`) seeded per replication.  
Baseline statistics in the paper are generated by running **multiple seeds** (default: 50) and summarizing outputs as mean ± SD across replications.

## Citation

If you use this code, please cite the software record (and the paper, if/when published):

Baker, P. C. (2026). *An Agent-Based Disruption Generator for Classrooms Using Lognormal Risk and Gamma–Poisson Counts* (Version v1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.18148012

## License

Add a license before wider distribution (common choices: MIT, BSD-3, Apache-2.0).  
If you do not include a license, others technically cannot reuse the code.
