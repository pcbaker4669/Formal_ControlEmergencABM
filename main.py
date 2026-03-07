
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

        # --- Control strength demo: same seed, vary control multiplier ---
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

        # --- New: single-run timeline plot for control ---
    p = Params(**model_params)
    result = simulate(p)

    plot_control_timeline(
        result["history"],
        threshold=p.control_threshold,
        outpath=os.path.join(figures_dir, "fig0_control_timeline_seed1.png"),
        title=f"Incidents, spikes, triggers, and control (seed={p.seed})"
    )



    # replicate_summaries(base_params)
    make_table1(model_params, seeds=range(1, 51), out_csv=os.path.join(data_dir, "table1_baseline.csv"))

    # Figure 1: Lorenz curve (student concentration)
    plot_lorenz_from_model(
        model_params,
        seeds=range(1, 51),
        out_png=os.path.join(figures_dir, "fig1_lorenz_concentration.png"),
        # out_pdf=os.path.join(figures_dir, "fig1_lorenz_concentration.pdf"),
    )

    # Figure 2
    plot_ccdf_class_counts(
        model_params,
        seeds=range(1, 51),
        out_png=os.path.join(figures_dir, "fig2_ccdf_class_counts.png"),
        #out_pdf=os.path.join(figures_dir, "fig2_ccdf_class_counts.pdf"),
        use_log_y=False,
    )

    tail_probabilities_class_counts(model_params, seeds=range(1, 51), thresholds=(10, 20, 30))

    # --- Sensitivity / robustness (OFAT) ---
    seeds = range(1, 51)

    # risk_sigma -> concentration
    rows_sigma = sweep_one_param(model_params, "risk_sigma", values=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0], seeds=seeds)
    plot_sweep(
        rows_sigma,
        y_key="top5_mean", yerr_key="top5_sd",
        title="Sensitivity: Concentration vs Risk dispersion",
        xlabel="Risk dispersion (risk_sigma)",
        ylabel="Top 5% share of incidents",
        out_png=os.path.join(figures_dir, "fig3_sweep_risk_sigma_top5.png"),
        #out_pdf=os.path.join(figures_dir, "fig3_sweep_risk_sigma_top5.pdf")
    )

    # nb_k -> burstiness (Var/Mean)
    rows_k = sweep_one_param(model_params, "nb_k", values=[0.2, 0.3, 0.5, 0.8, 1.0, 1.5], seeds=seeds)
    plot_sweep(
        rows_k,
        y_key="varmean_mean", yerr_key="varmean_sd",
        title="Sensitivity: Overdispersion vs Burstiness",
        xlabel="Burstiness (nb_k)",
        ylabel="Var/Mean of class-period counts",
        out_png=os.path.join(figures_dir, "fig4_sweep_nb_k_varmean.png"),
        # out_pdf=os.path.join(figures_dir, "fig4_sweep_nb_k_varmean.pdf")
    )


    # inc_base_rate -> level (class_mean)
    rows_rate = sweep_one_param(model_params, "inc_base_rate", values=[0.12, 0.18, 0.24, 0.30, 0.36], seeds=seeds)
    plot_sweep(
        rows_rate,
        y_key="class_mean_mean", yerr_key="class_mean_sd",
        title="Sensitivity: Incident level vs Baseline incident rate",
        xlabel="Baseline incident rate (inc_base_rate)",
        ylabel="Mean incidents per class-period",
        out_png=os.path.join(figures_dir, "fig5_sweep_inc_base_rate_level.png"),
        # out_pdf=os.path.join(figures_dir, "fig5_sweep_inc_base_rate_level.pdf")
    )


if __name__ == "__main__":
    main()
