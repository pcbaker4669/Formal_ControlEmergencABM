
import json
import os

from analysis import (
    make_table1,
    plot_lorenz_from_model,
    plot_ccdf_class_counts,
    tail_probabilities_class_counts,
    sweep_one_param,
    plot_sweep,
    plot_control_timeline,
    plot_lognormal_distribution,
    plot_gamma_distribution,
    plot_poisson_distribution,
)

from model_core import Params, simulate

# ----------------------------
# Main
# ----------------------------
def main():
    # multiple runs with different seeds
    with open("config.json", "r") as f:
        base_params = json.load(f)

    figures_dir = base_params.get("figures_dir", "figures")
    data_dir = base_params.get("data_dir", "data")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    model_params = {
        k: v for k, v in base_params.items()
        if k not in ("figures_dir", "data_dir")
    }

    plot_lognormal_distribution(
        mu=model_params["risk_mu"],
        sigma=model_params["risk_sigma"],
        out_png=os.path.join(figures_dir, "fig_lognormal.png"),
    )

    plot_gamma_distribution(
        k=model_params["nb_k"],
        theta=1.0,
        out_png=os.path.join(figures_dir, "fig_gamma.png"),
    )

    plot_poisson_distribution(
        lam=3.0,
        max_k=10,
        out_png=os.path.join(figures_dir, "fig_poisson.png"),
    )

    # --- Control strength demo: varying control_multiplier (0.5, 0.75, 1.0) with fixed duration=4, using baseline parameters otherwise ---
    base_for_timeline = dict(model_params)
    base_for_timeline["control_duration_days"] = 4



    for mult in (1.0, 0.75, 0.5):
        p = Params(**{**base_for_timeline, "control_multiplier": mult, "seed": 1})
        result = simulate(p)
        tag = str(mult).replace(".", "_")
        plot_control_timeline(
            result["history"],
            threshold=p.control_threshold,
            outpath=os.path.join(figures_dir, f"fig0_control_timeline_mult_{tag}.png"),
            title=f"Control strength demo (mult={mult}, duration=4, seed=1)"
        )

    # --- Single-run timeline plot: using baseline parameters from config.json ---
    p = Params(**model_params)
    result = simulate(p)

    plot_control_timeline(
        result["history"],
        threshold=p.control_threshold,
        outpath=os.path.join(figures_dir, "fig0_control_timeline_seed1.png"),
        title=f"Incidents, spikes, triggers, and control (seed={p.seed})"
    )



    # replicate_summaries(base_params)
    # Table 1: Baseline summary statistics using parameters from config.json
    make_table1(model_params, seeds=range(1, 51), out_csv=os.path.join(data_dir, "table1_baseline.csv"))

    # Figure 1: Lorenz curve (student concentration) using baseline parameters from config.json
    plot_lorenz_from_model(
        model_params,
        seeds=range(1, 51),
        out_png=os.path.join(figures_dir, "fig1_lorenz_concentration.png"),
        out_pdf=None,
    )

    # Figure 2: CCDF of class incident counts using baseline parameters from config.json
    plot_ccdf_class_counts(
        model_params,
        seeds=range(1, 51),
        out_png=os.path.join(figures_dir, "fig2_ccdf_class_counts.png"),
        out_pdf=None,
    )

    # Tail probabilities: P(class incidents > threshold) using baseline parameters from config.json
    tail_probabilities_class_counts(model_params, seeds=range(1, 51), thresholds=(10, 20, 30))

    # --- Sensitivity / robustness (OFAT) using baseline parameters from config.json ---

    # Figure 3: Sensitivity of concentration (top 5% share) to risk dispersion (risk_sigma), varying risk_sigma values
    seeds = range(1, 51)
    rows_sigma = sweep_one_param(
        model_params, "risk_sigma", 
        values=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0], seeds=seeds)
    plot_sweep(
        rows_sigma,
        y_key="top5_mean", yerr_key="top5_sd",
        title="Sensitivity: Concentration vs Risk dispersion",
        xlabel="Risk dispersion (risk_sigma)",
        ylabel="Top 5% share of incidents",
        out_png=os.path.join(figures_dir, "fig3_sweep_risk_sigma_top5.png"),
        out_pdf=None,
    )

    # Figure 4: Sensitivity of overdispersion (Var/Mean) to burstiness (nb_k), varying nb_k values
    rows_k = sweep_one_param(model_params, "nb_k", values=[0.2, 0.3, 0.5, 0.8, 1.0, 1.5], seeds=seeds)
    plot_sweep(
        rows_k,
        y_key="varmean_mean", yerr_key="varmean_sd",
        title="Sensitivity: Overdispersion vs Burstiness",
        xlabel="Burstiness (nb_k)",
        ylabel="Var/Mean of class-period counts",
        out_png=os.path.join(figures_dir, "fig4_sweep_nb_k_varmean.png"),
        out_pdf=None,
    )

    # Figure 5: Sensitivity of incident level (class_mean) to baseline rate (inc_base_rate), varying inc_base_rate values
    rows_rate = sweep_one_param(model_params, "inc_base_rate", values=[0.12, 0.18, 0.24, 0.30, 0.36], seeds=seeds)
    plot_sweep(
        rows_rate,
        y_key="class_mean_mean", yerr_key="class_mean_sd",
        title="Sensitivity: Incident level vs Baseline incident rate",
        xlabel="Baseline incident rate (inc_base_rate)",
        ylabel="Mean incidents per class-period",
        out_png=os.path.join(figures_dir, "fig5_sweep_inc_base_rate_level.png"),
        out_pdf=None,
    )



if __name__ == "__main__":
    main()
