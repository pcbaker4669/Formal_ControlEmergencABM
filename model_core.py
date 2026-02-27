"""Core model components for the classroom disruption generator.

Inputs: a Params instance (seed, student count, class size, rates, and days).
Outputs: history (daily totals), class_day_records (per-class/day rows),
students (per-student totals), and a summary dict of aggregate metrics.
Public API: Params, Student, Model, simulate; summary keys are stable outputs.
Basic use: call simulate(params) for a one-shot run or Model(params).run().
"""

from dataclasses import dataclass
import numpy as np
import math

# public API for from module import *
__all__ = ["Params", "Student", "Model", "simulate"]

# ----------------------------
# Parameters (model-specific)
# ----------------------------
@dataclass
class Params:
    seed: int = 1
    n_students: int = 300
    n_days: int = 90

    # Class structure
    class_size: int = 30  # fixed size (you can generalize later)

    # Disruption process, this scales class_mean (incident rate) linearly
    # 0.02 -> class_mean ~ 0.83,
    # if you want class_mean ~ 5 then .2 * (5/0.83) ~ 0.12
    # 0.24 -> class_mean ~ 10, 0.36 -> class_mean ~ .15
    inc_base_rate: float = 0.02   # baseline expected incidents per student per day (before modifiers)

    # nb_k is the shape parameter (k) that forces the model to follow a Negative Binomial 
    # (nb) distribution, thereby producing the specific "burstiness" required to match 
    # empirical classroom data.
    nb_k: float = 0.5           # dispersion; smaller => heavier tail, (originally 1)
                                # .5 - moderate burstiness
                                # .2 - strong burstiness

    # Latent risk distribution for students
    # risk_mu to hit your target average system load (total incidents/day).
    risk_mu: float = -0.78          # lognormal mean (in log space, originally 0)

    # Use risk_sigma to hit your target concentration (top 5% share ~.3 Originally 1).
    # top 5% share ~.45 with value of 1.6
    risk_sigma: float = 1.6       # lognormal sigma (bigger => more concentration)

    # for literature-style subgroup comparisons (e.g., "at-risk" students)
    at_risk_top_n: int = 3


# ----------------------------
# Student agent
# ----------------------------
class Student:
    def __init__(self, sid: int, risk: float):
        self.sid = sid
        self.risk = float(risk)
        self.incidents_total = 0

# ----------------------------
# Model
# ----------------------------
class Model:
    def __init__(self, p: Params):
        self.p = p
        self.rng = np.random.default_rng(p.seed)

        self.students = self._init_students()
        self.classes = self._init_classes()

        # map student -> class_id (useful for tables)
        self.class_of = {}
        for cid, cls in enumerate(self.classes):
            for sid in cls:
                self.class_of[int(sid)] = cid

        # define "at-risk" as top-N risk students per class (fixed for the run)
        self.at_risk_by_class = []
        for cls in self.classes:
            cls_ids = np.array([int(sid) for sid in cls])
            cls_risks = np.array([self.students[sid].risk for sid in cls_ids])
            top_idx = np.argsort(cls_risks)[::-1][: self.p.at_risk_top_n]
            self.at_risk_by_class.append(set(cls_ids[top_idx].tolist()))

        self.history = []             # one row per day (overall totals)
        self.class_day_records = []   # one row per class per day (literature-aligned)

    def _init_students(self):
        # Latent disruption risk (positive, heavy-tailed)
        risk = self.rng.lognormal(mean=self.p.risk_mu, sigma=self.p.risk_sigma, size=self.p.n_students)
        return [Student(sid=i, risk=risk[i]) for i in range(self.p.n_students)]

    def _init_classes(self) -> list[np.ndarray]:
        # Partition students into fixed classes (rosters).
        order = self.rng.permutation(self.p.n_students)
        return [order[i:i + self.p.class_size] for i in range(0, self.p.n_students, self.p.class_size)]

    def step_day(self, day: int):
        total_incidents_today = 0

        for cid, cls in enumerate(self.classes):
            class_total = 0
            at_risk_total = 0

            at_risk_set = self.at_risk_by_class[cid]

            for sid in cls:
                s = self.students[int(sid)]
                lam = self.p.inc_base_rate * s.risk

                k = max(self.p.nb_k, 1e-12)
                # Draw a latent incident rate lam_tilde from a Gamma distribution to model 
                # unobserved day-to-day volatility (so the rate itself can fluctuate).
                lam_tilde = self.rng.gamma(shape=k, scale=lam / k if lam > 0 else 0.0)
                # Draw the realized incident count k_i from a Poisson distribution using 
                # lam_tilde as the rate, turning that fluctuating rate into actual bursty counts.
                # Poisson distribution predicts the probability of k events occurring within that interval
                k_i = int(self.rng.poisson(lam_tilde))
                s.incidents_total += k_i
                class_total += k_i
                total_incidents_today += k_i

                if int(sid) in at_risk_set:
                    at_risk_total += k_i

            self.class_day_records.append({
                "day": day,
                "class_id": cid,
                "class_size": len(cls),
                "incidents_class": class_total,
                "incidents_at_risk": at_risk_total,
                "incidents_nonrisk": class_total - at_risk_total,
                "at_risk_n": len(at_risk_set),
            })

        self.history.append({"day": day, "incidents_total": total_incidents_today})

    def run(self):
        for day in range(1, self.p.n_days + 1):
            self.step_day(day)
        return self.history

    @staticmethod
    def share_top(x, frac):
        x = np.asarray(x, dtype=float)
        s = x.sum()
        if s <= 0:
            return 0.0
        k = max(1, int(math.ceil(frac * len(x))))
        return np.sort(x)[::-1][:k].sum() / s

    def student_table(self):
        rows = []
        for s in self.students:
            rows.append({
                "sid": s.sid,
                "class_id": self.class_of[s.sid],
                "risk": s.risk,
                "incidents_total": s.incidents_total,
                "incidents_per_day": s.incidents_total / self.p.n_days,
            })
        return rows


    def summary(self):
        daily = np.array([h["incidents_total"] for h in self.history], dtype=float)

        student_totals = np.array([s.incidents_total for s in self.students], dtype=float)

        class_totals = np.array([r["incidents_class"] for r in self.class_day_records], dtype=float)
        at_risk_totals = np.array([r["incidents_at_risk"] for r in self.class_day_records], dtype=float)

        # per at-risk student per class-period (top N per class)
        at_risk_n = float(self.p.at_risk_top_n)
        at_risk_per_student = at_risk_totals / max(at_risk_n, 1.0)

        out = {
            "daily_mean": float(daily.mean()),
            "daily_var_mean": float(daily.var(ddof=1) / max(daily.mean(), 1e-12)),

            "class_mean": float(class_totals.mean()),
            "class_var_mean": float(class_totals.var(ddof=1) / max(class_totals.mean(), 1e-12)),
            "class_zero_frac": float((class_totals == 0).mean()),
            "class_q50": float(np.quantile(class_totals, 0.50)),
            "class_q90": float(np.quantile(class_totals, 0.90)),
            "class_q95": float(np.quantile(class_totals, 0.95)),
            "class_q99": float(np.quantile(class_totals, 0.99)),
            "class_max": float(class_totals.max()),

            "at_risk_class_mean": float(at_risk_totals.mean()),
            "at_risk_per_student_mean": float(at_risk_per_student.mean()),
            "at_risk_share_total": float(at_risk_totals.sum() / max(class_totals.sum(), 1e-12)),

            "top5_share_students": float(self.share_top(student_totals, 0.05)),
            "top1_share_students": float(self.share_top(student_totals, 0.01)),
            "student_zero_frac": float((student_totals == 0).mean()),
        }
        return out
    

def simulate(params: Params) -> dict:
    """Run the model and return core outputs in a structured dict."""
    m = Model(params)
    m.run()
    return {
        "history": m.history,
        "class_day_records": m.class_day_records,
        "students": m.student_table(),
        "summary": m.summary(),
    }
