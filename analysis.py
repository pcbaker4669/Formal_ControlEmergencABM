"""Analysis and plotting utilities for the polity disruption model."""

from model_core import Model, Params, simulate
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from scipy.stats import lognorm, gamma, poisson


__all__ = [
    "make_table1",
    "plot_lorenz_from_model",
    "plot_ccdf_polity_counts",
    "tail_probabilities_polity_counts",
    "sweep_one_param",
    "plot_sweep",
]

def plot_control_timeline(history, threshold=None, outpath=None, title=None):
    days = np.array([h["day"] for h in history], dtype=int)
    incidents = np.array([h["incidents_total"] for h in history], dtype=float)
    triggered = np.array([h["control_triggered_today"] for h in history], dtype=bool)
    control_on = np.array([h["control_active"] for h in history], dtype=bool)

    fig, ax = plt.subplots()
    ax.plot(days, incidents, marker="o", linewidth=1)

    # optional threshold line
    if threshold is not None:
        ax.axhline(threshold, linestyle="--")

    # mark trigger days (vertical lines)
    for d in days[triggered]:
        ax.axvline(d, linestyle=":", linewidth=1)

    # shade control-active days (light vertical bands)
    for d in days[control_on]:
        ax.axvspan(d - 0.5, d + 0.5, alpha=0.15)

    ax.set_xlabel("Day")
    ax.set_ylabel("Total incidents")
    ax.set_title(title or "Incidents over time with triggers and control days")

    # simple legend proxy
    # (matplotlib doesn't auto-legend axvline/axvspan nicely without handles)
    ax.text(0.01, 0.98,
            "Dotted line = trigger day\nShaded = control active",
            transform=ax.transAxes, va="top")

    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
    else:
        plt.show()

def run_sweep():
    for rate in [0.12, 0.24, 0.36]:
        p = Params(inc_base_rate=rate)
        out = simulate(p)
        s = out["summary"]

        print("\ninc_base_rate =", rate)
        for k in ["polity_mean","polity_q50","polity_q90","polity_q95","polity_q99","polity_max",
                  "polity_zero_frac","polity_var_mean",
                  "at_risk_share_total","top5_share_students","student_zero_frac"]:
            print(k, "=", round(float(s[k]), 4))

def replicate_summaries(base_params, seeds=range(1, 51)):
    keys = [
        "polity_mean","polity_q50","polity_q90","polity_q95","polity_q99","polity_max",
        "polity_zero_frac","polity_var_mean",
        "at_risk_share_total","top5_share_students","student_zero_frac"
    ]
    rows = []
    for sd in seeds:
        p = Params(**{**base_params, "seed": sd})
        out = simulate(p)
        s = out["summary"]
        rows.append([float(s[k]) for k in keys])

    arr = np.array(rows, dtype=float)
    print("Replications:", len(seeds))
    for i, k in enumerate(keys):
        mean = arr[:, i].mean()
        sd = arr[:, i].std(ddof=1)
        print(f"{k}: mean={mean:.3f} sd={sd:.3f}")

def make_table1(base_params: dict, seeds=range(1, 51), out_csv="table1_baseline.csv"):
    """
    Runs replications over seeds and produces Table 1 (mean ± sd across runs).
    Saves per-run summaries and Table 1 to CSV.
    """
    keys = [
        "polity_mean","polity_q50","polity_q90","polity_q95","polity_q99","polity_max",
        "polity_zero_frac","polity_var_mean",
        "at_risk_share_total","top5_share_students","student_zero_frac",
        "control_days_total", "control_activation_count", "control_day_frac", 
        "trigger_day_frac", "incidents_mean_control_on",
        "incidents_mean_control_off", "trigger_day_mean_incidents", "next_day_after_trigger_mean_incidents",
        "mean_drop_after_trigger"
    ]

    # Collect per-run rows
    rows = []
    for sd in seeds:
        p = Params(**{**base_params, "seed": sd})
        out = simulate(p)
        s = out["summary"]
        row = {}
        for k in keys:
            if k in ("control_days_total", "control_activation_count"):
                row[k] = int(s[k])
            else:
                row[k] = float(s[k])
                row["seed"] = sd
                rows.append(row)

    # Build numeric array for mean/sd
    X = np.array([[r[k] for k in keys] for r in rows], dtype=float)
    means = X.mean(axis=0)
    sds = X.std(axis=0, ddof=1)

    # Print Table 1
    print("\nTABLE 1 (Baseline generator outputs across replications)")
    print(f"Replications: {len(list(seeds))}")
    for k, mu, sd in zip(keys, means, sds):
        print(f"{k}: mean={mu:.3f} sd={sd:.3f}")

    # Save per-run summaries
    per_run_csv = out_csv.replace(".csv", "_per_run.csv")
    with open(per_run_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed"] + keys)
        w.writeheader()
        w.writerows(rows)

    # Save Table 1 (mean/sd) as a simple CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "sd"])
        for k, mu, sd in zip(keys, means, sds):
            w.writerow([k, float(mu), float(sd)])

    print(f"\nWrote per-run summaries to: {per_run_csv}")
    print(f"Wrote Table 1 to: {out_csv}")

    return {k: (float(mu), float(sd)) for k, mu, sd in zip(keys, means, sds)}

def pooled_tail_prob_polity_counts(base_params: dict, seeds, threshold: int) -> float:
    pooled = []
    for sd in seeds:
        p = Params(**{**base_params, "seed": sd})
        out = simulate(p)
        pooled.extend([r["incidents_polity"] for r in out["polity_day_records"]])

    pooled = np.array(pooled, dtype=int)
    return float((pooled >= threshold).mean())


def sweep_one_param(base_params: dict, param_name: str, values, seeds=range(1, 51), tail_threshold=20):
    """
    OFAT sweep: vary one parameter, replicate across seeds, return mean±sd for key outputs.
    """
    values = list(values)
    seeds = list(seeds)

    rows = []
    for v in values:
        per_run = []
        for sd in seeds:
            p = Params(**{**base_params, "seed": sd, param_name: v})
            out = simulate(p)
            per_run.append(out["summary"])
        # summarize across runs (seed replications)
        def mu_sd(key):
            arr = np.array([r[key] for r in per_run], dtype=float)
            return float(arr.mean()), float(arr.std(ddof=1))

        out = {"param": param_name, "value": float(v)}
        # concentration
        out["top5_mean"], out["top5_sd"] = mu_sd("top5_share_students")
        out["top1_mean"], out["top1_sd"] = mu_sd("top1_share_students")
        # burstiness + level
        out["polity_mean_mean"], out["polity_mean_sd"] = mu_sd("polity_mean")
        out["varmean_mean"], out["varmean_sd"] = mu_sd("polity_var_mean")
        # pooled tail probability (more stable than per-run tails)
        out["p_tail"] = pooled_tail_prob_polity_counts({**base_params, param_name: v}, seeds, tail_threshold)

        rows.append(out)

    return rows


def plot_sweep(rows, x_key="value", y_key="top5_mean", yerr_key="top5_sd",
               title="", xlabel="", ylabel="",
               out_png="sweep.png", out_pdf="sweep.pdf"):
    x = np.array([r[x_key] for r in rows], dtype=float)
    y = np.array([r[y_key] for r in rows], dtype=float)
    yerr = np.array([r[yerr_key] for r in rows], dtype=float) if yerr_key else None

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
    })

    fig = plt.figure(figsize=(6.5, 5.0))
    ax = plt.gca()

    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3)
    else:
        ax.plot(x, y, marker="o")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if out_png:
        plt.savefig(out_png)

    if out_pdf:
        plt.savefig(out_pdf)

    plt.close(fig)
    print("Saved:", out_png, "and", out_pdf)


def share_top(x, frac):
    x = np.asarray(x, dtype=float)
    s = x.sum()
    if s <= 0:
        return 0.0
    k = max(1, int(math.ceil(frac * len(x))))
    return float(np.sort(x)[::-1][:k].sum() / s)

def lorenz_curve(values):
    v = np.asarray(values, dtype=float)
    v = np.maximum(v, 0.0)
    v_sorted = np.sort(v)
    cum = np.cumsum(v_sorted)
    total = cum[-1] if len(cum) else 0.0
    if total <= 0:
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        return x, y, 0.0

    x = np.concatenate([[0.0], np.arange(1, len(v_sorted) + 1) / len(v_sorted)])
    y = np.concatenate([[0.0], cum / total])

    area = np.trapezoid(y, x)
    gini = 1.0 - 2.0 * area
    return x, y, float(gini)

def plot_lorenz_from_model(
    base_params: dict,
    seeds=range(1, 51),
    out_png="fig1_lorenz_concentration.png",
    out_pdf="fig1_lorenz_concentration.pdf",
):
    """
    Runs the ABM across multiple seeds, pools student incident totals, and saves
    a journal-quality Lorenz curve figure (PNG + PDF).
    """
    seeds = list(seeds)

    pooled_totals = []
    top5_shares = []
    top1_shares = []

    for sd in seeds:
        p = Params(**{**base_params, "seed": sd})
        out = simulate(p)
        totals = np.array([s["incidents_total"] for s in out["students"]], dtype=float)

        pooled_totals.append(totals)
        top5_shares.append(share_top(totals, 0.05))
        top1_shares.append(share_top(totals, 0.01))

    pooled_totals = np.concatenate(pooled_totals)
    x, y, gini = lorenz_curve(pooled_totals)

    top5_mean = float(np.mean(top5_shares))
    top1_mean = float(np.mean(top1_shares))

    # Journal-ish matplotlib defaults (no explicit colors)
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
    })

    fig = plt.figure(figsize=(6.5, 5.0))
    ax = plt.gca()

    ax.plot(x, y, label="Lorenz curve\n(pooled student totals)")
    ax.plot([0, 1], [0, 1], linestyle="--", label="_nolegend_")

    ax.set_title("Concentration of Incidents Across Students (Lorenz Curve)")
    ax.set_xlabel("Cumulative share of students")
    ax.set_ylabel("Cumulative share of incidents")

    txt = (
        f"Pooled over {len(seeds)} runs\n(N={len(pooled_totals)} student-runs)\n\n"
        f"Gini = {gini:.3f}\n"
        f"Mean top 5% share = {top5_mean:.3f}\nmean top 1% share = {top1_mean:.3f}"
    )
    ax.text(0.05, 0.7, txt, transform=ax.transAxes)
    ax.text(
        0.5, 0.53,
        "Equality line",
        transform=ax.transAxes,
        rotation=36,
        rotation_mode="anchor",
        ha="center",
        va="center",
        fontsize=10,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if out_png:
        plt.savefig(out_png)

    if out_pdf:
        plt.savefig(out_pdf)

    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def plot_ccdf_polity_counts(
    base_params: dict,
    seeds=range(1, 51),
    out_png="fig2_ccdf_polity_counts.png",
    out_pdf="fig2_ccdf_polity_counts.pdf",
    use_log_y=True,
):
    """
    Figure 2: CCDF (survival function) of polity-period incident counts.
    Pools incidents_polity across all polity-periods and replications.
    Saves PNG + PDF.
    """
    seeds = list(seeds)

    pooled_counts = []

    for sd in seeds:
        p = Params(**{**base_params, "seed": sd})
   
        out = simulate(p)
        # one count per polity per day
        counts = np.array([r["incidents_polity"] for r in out["polity_day_records"]], 
                          dtype=float)
        pooled_counts.append(counts)

    pooled_counts = np.concatenate(pooled_counts)
    # data-driven cap (e.g., 99.5th percentile)
    xmax = int(np.quantile(pooled_counts, 0.995))

    # CCDF: P(X >= x)
    x = np.sort(pooled_counts)
    n = len(x)
    y = 1.0 - (np.arange(1, n + 1) / n)  # survival

    # Avoid y=0 on log scale
    y = np.maximum(y, 1.0 / n)

    # --- IMPORTANT: truncate the *plotted data* to match your x-axis truncation ---
    mask = x <= xmax
    x_plot = x[mask]
    y_plot = y[mask]

    # Journal-ish matplotlib defaults (same approach as Fig 1)
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
    })

    fig = plt.figure(figsize=(6.5, 5.0))
    ax = plt.gca()

    ax.set_xlim(0, xmax)
    ax.text(
        0.62, 0.92,  # (x,y) in axes coords
        f"x-axis truncated at {xmax} (99.5th pct)",
        transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2)
    )

    ax.step(x_plot, y_plot, where="post", label="CCDF of polity-period incidents")
    ax.set_xlim(0, xmax)

    # Now set y-limits based on what you're actually plotting
    if use_log_y:
        ymin = y_plot.min()
        y_floor = 10 ** np.floor(np.log10(ymin))  # nice decade floor (e.g., 1e-3)
        ax.set_ylim(y_floor, 1.0)
        ax.set_yscale("log")
    else:
        ax.set_ylim(0.0, 1.0)

    ax.set_title("Polity-Period Incident Counts (CCDF)")
    ax.set_xlabel("Incidents per polity-period")
    ax.set_ylabel("P(X >= x)")

    if use_log_y:
        ax.set_yscale("log")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False)

    plt.tight_layout()

    if out_png:
        plt.savefig(out_png)

    if out_pdf:
        plt.savefig(out_pdf)

    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

def tail_probabilities_polity_counts(base_params: dict, seeds=range(1, 51), thresholds=(10, 20, 30)):
    seeds = list(seeds)
    pooled = []

    for sd in seeds:
        p = Params(**{**base_params, "seed": sd})
        out = simulate(p)
        pooled.append([r["incidents_polity"] for r in out["polity_day_records"]])

    pooled = np.array([x for sub in pooled for x in sub], dtype=int)

    print("\nTail probabilities for polity-period incident counts (pooled):")
    print(f"P(X = 0) = {(pooled == 0).mean():.3f}")
    for t in thresholds:
        print(f"P(X >= {t}) = {(pooled >= t).mean():.3f}")


def lognormal_pdf(x, mu, sigma):
    return (1.0 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(x) - mu) ** 2) / (2 * sigma ** 2)
    )


def gamma_pdf(x, k, theta):
    return (x ** (k - 1) * np.exp(-x / theta)) / (math.gamma(k) * (theta ** k))


def poisson_pmf(k_vals, lam):
    k_vals = np.asarray(k_vals, dtype=int)
    return np.array([
        (lam ** k) * np.exp(-lam) / math.factorial(int(k))
        for k in k_vals
    ])

def plot_lognormal_distribution(mu=-0.78, sigma=1.6, out_png="figures/fig_lognormal.png"):
    x = np.linspace(0.001, 2.5, 500)
    y = lognormal_pdf(x, mu, sigma)

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.0,
    })

    fig = plt.figure(figsize=(6.5, 4.5))
    ax = plt.gca()

    
    ax.set_xlim(0, 2.5)

    # Note: The parameter mu describes the center of the distribution in log-space. Because 
    # the lognormal is right-skewed, the actual average on the original scale is larger; 
    # this is called the raw mean.
    raw_mean = np.exp(mu + 0.5 * sigma**2)
    ax.plot(x, y, label=f"μ = {mu:.2f}\nraw μ = {raw_mean:.2f}\nσ = {sigma:.2f}")
    ax.legend(frameon=False, loc="upper right")

    ax.set_title("Lognormal Distribution")

    ax.set_xlabel("Risk")
    ax.set_ylabel("Density")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if out_png:
        plt.savefig(out_png)
        print(f"Saved: {out_png}")
    plt.close(fig)

def plot_gamma_distribution(k=0.5, theta=1.0, out_png="figures/fig_gamma.png"):
    x = np.linspace(0.001, 8, 500)
    y = gamma_pdf(x, k, theta)

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.0,
    })

    fig = plt.figure(figsize=(6.5, 4.5))
    ax = plt.gca()
    mean_val = k * theta

    ax.plot(x, y, label=f"k = {k:.2f}, θ = {theta:.2f}\nμ = {mean_val:.2f}")
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 3)
    ax.set_title("Gamma Distribution (Latent Rate Volatility)") 
    ax.text(
        0.98, 0.85,
        "Rate variability\nbefore Poisson events",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9
    )
    ax.set_xlabel("Latent Rate")
    ax.set_ylabel("Density")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)



    plt.tight_layout()
    if out_png:
        plt.savefig(out_png)
        print(f"Saved: {out_png}")
    plt.close(fig)


def plot_poisson_distribution(lam=3.0, max_k=10, out_png="figures/fig_poisson.png"):
    k_vals = np.arange(0, max_k + 1)
    y = poisson_pmf(k_vals, lam)

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.0,
    })

    fig = plt.figure(figsize=(6.5, 4.5))
    ax = plt.gca()

    markerline, stemlines, baseline = ax.stem(
        k_vals, y,
        label=f"λ = {lam:.2f}\nexpected incidents per period"
    )
    baseline.set_visible(False)
    plt.setp(markerline, markersize=5)
    plt.setp(stemlines, linewidth=1.5)

    ax.set_title("Poisson Event Count Distribution")
    ax.set_xlabel("Incidents in a polity-period")
    ax.set_ylabel("Probability")
    ax.set_xlim(-0.5, max_k + 0.5)
    ax.set_ylim(0, max(y) * 1.12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    if out_png:
        plt.savefig(out_png)
        print(f"Saved: {out_png}")
    plt.close(fig)


# The Negative Binomial PMF (Probability Mass Function) gives the probability 
# that a Negative Binomial random variable takes a specific integer value.
def negbin_pmf(x_vals, r, mean_):
    """
    Negative Binomial PMF parameterized by:
    r = dispersion/shape
    mean_ = desired mean

    Uses:
        p = r / (r + mean)
        P(X=x) = C(x+r-1, x) * p^r * (1-p)^x
    """
    p = r / (r + mean_)
    out = []
    for x in x_vals:
        coeff = math.gamma(x + r) / (math.gamma(r) * math.factorial(int(x)))
        pmf = coeff * (p ** r) * ((1 - p) ** x)
        out.append(pmf)
    return np.array(out, dtype=float)

# The Gamma–Poisson mixture implies a Negative Binomial count process, producing bursty, 
# overdispersed event patterns consistent with empirical disruption data.
def plot_negative_binomial_distribution(
    r=0.5,
    mean_=3.0,
    max_x=20,
    out_png="figures/fig_negative_binomial.png"
):
    x_vals = np.arange(0, max_x + 1)
    y = negbin_pmf(x_vals, r, mean_)

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.0,
    })

    fig = plt.figure(figsize=(6.5, 4.5))
    ax = plt.gca()

    markerline, stemlines, baseline = ax.stem(x_vals, y)
    baseline.set_visible(False)
    plt.setp(markerline, markersize=5)
    plt.setp(stemlines, linewidth=1.5)

    ax.set_title("Negative Binomial Count Distribution")
    ax.set_xlabel("Incidents")
    ax.set_ylabel("Probability")
    ax.set_xlim(-0.5, max_x + 0.5)
    ax.set_ylim(0, max(y) * 1.12)
    ax.set_xticks(np.arange(0, max_x + 1, 2))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if out_png:
        plt.savefig(out_png)
        print(f"Saved: {out_png}")
    plt.close(fig)