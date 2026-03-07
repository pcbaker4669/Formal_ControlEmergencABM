# Disorder, Tail Risk, and the Emergence of Control (Toy ABM)

This project is a small, presentation-ready agent-based modeling (ABM) sandbox built from a **classroom disruption generator** and adapted for an *Origins of Social Complexity* framing:

- **Disorder is concentrated**: a small fraction of individuals can generate a large share of incidents.
- **Disorder is bursty**: long quiet stretches punctuated by spikes (tail events).
- **Control emerges as a response**: when incidents exceed a threshold, the model enters a temporary “tight control” regime that reduces incident rates.

The model is intentionally simple (toy-scale) and designed for clear, reproducible figures rather than realism.

---

## Project structure

- `model_core.py`  
  Core ABM: parameters, student agents, population growth schedule, disruption process (Gamma–Poisson / Negative Binomial mixture), and the control rule.
- `analysis.py`  
  Analysis + figure generation helpers (tables, Lorenz curve, CCDF, parameter sweeps, and the control timeline plot).
- `main.py`  
  Orchestrates runs using `config.json`, writes tables, and generates figures.
- `config.json`  
  Default run configuration (model parameters + output directories).
- `data/` (generated)  
  CSV outputs (e.g., Table 1 runs across seeds).
- `figures/` (generated)  
  Saved plots (PNG/PDF).

---

## Model overview

### Population growth (scalar stress proxy)
The simulation runs for `n_days` (default 90). The active population grows in stages:

- Days 1–30: `pop_stage_1` (e.g., 30)
- Days 31–60: `pop_stage_2` (e.g., 60)
- Days 61–90: `pop_stage_3` (e.g., 100)

All agents exist from the start, but only the first `active_n` are “active” on a given day (toy simplification).

### Disruption generation (bursty counts)
Each individual has a fixed latent risk drawn from a lognormal distribution (`risk_mu`, `risk_sigma`) that controls inequality/concentration.

Daily incident counts are generated using a Gamma–Poisson mixture:
- Draw a volatile daily rate `lam_tilde ~ Gamma(k, scale=lam/k)`
- Draw realized incidents `k_i ~ Poisson(lam_tilde)`

This yields overdispersed (bursty) counts consistent with Negative Binomial behavior.

### Control rule (emergence of formal control)
If total incidents on a day exceed `control_threshold`, the model activates “control” for `control_duration_days` starting the **next day**.
During control, individual incident rates are multiplied by `control_multiplier` (e.g., 0.75 or 0.5).

The daily history records:
- `incidents_total`
- `control_active`
- `control_triggered_today`

The summary includes:
- control activations, share of control days
- average incidents under control vs not
- mean change from trigger day to the next day (effect proxy)

---

## Quick start

### 1) Create and activate a virtual environment (recommended)

**Windows (PowerShell):**
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install requirements
```bash
pip install -r requirements.txt
```
If you don’t have a `requirements.txt`, these are typically sufficient:
```bash
pip install numpy matplotlib
```

### 3) Configure the run
Edit `config.json` to adjust the model:
- `n_students`, `n_days`, `class_size`
- `pop_stage_1`, `pop_stage_2`, `pop_stage_3`
- `inc_base_rate`, `nb_k`, `risk_mu`, `risk_sigma`
- `control_threshold`, `control_multiplier`, `control_duration_days`
- `figures_dir`, `data_dir`

### 4) Run everything
```bash
python main.py
```

This generates:
- `data/table1_baseline.csv` (multi-seed summary table)
- `figures/` plots (Lorenz, CCDF, sweeps, and control timeline if enabled in `main.py`)

---

## Suggested “demo” runs for class

### Control strength comparison (same seed)
Keep:
- `control_duration_days = 4`
Compare:
- `control_multiplier = 1.0` (placebo)
- `control_multiplier = 0.75` (mild)
- `control_multiplier = 0.5` (strong)

Then generate three timeline plots to visually show spikes, triggers, and the post-trigger regime.

---

## Outputs you’ll likely use in slides

- **Timeline plot:** incidents over time with trigger markers + shaded control days  
  (from `plot_control_timeline` in `analysis.py`)
- **Lorenz curve:** concentration of incidents across individuals
- **CCDF / tail probabilities:** frequency of extreme class/day incident counts
- **Table 1:** multi-seed summary statistics (means, quantiles, concentration)

---

## Reproducibility

- Randomness is controlled by `seed`.
- Many outputs in `main.py` run across `seeds=range(1, 51)` by default.
- For “clean comparisons,” keep the seed fixed and vary one parameter at a time.

---

## Notes / limitations (intentional)

This is a **toy model**. It abstracts away many real mechanisms (learning, adaptation, targeting, legitimacy costs, spatial spillovers, etc.) to focus on a single conceptual point:

> concentrated + bursty disorder creates pressure for formal control, especially as population scale increases.

---

## License
Use freely for coursework and teaching materials. Add a license if you plan to publish or distribute widely.
