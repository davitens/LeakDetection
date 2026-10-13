"""Probabilistic and cost-sensitive scoring rules.

LogLoss and Brier Score are proper scoring rules on the predicted probability.
The AUCC integrates the minimum expected cost over the operational cost space,
where the cost ratio ``R = C_FN / C_FP`` is mapped onto a normalised
probability cost

    Pc = p * R / (p * R + (1 - p))

with ``p`` the observed prevalence of leaks. Equations follow the manuscript's
"Scoring rules" section.
"""

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

#: Upper bound of the swept cost ratio. The manuscript states 10**3.2, i.e.
#: C_FN = 1654 * C_FP, the ratio at which flagging every pipe becomes optimal.
#: Table 2 and Figure 1 of the published paper were computed with 10**2.5; pass
#: that value explicitly to reproduce those numbers.
DEFAULT_MAX_COST_RATIO = 10**3.2

N_COST_STEPS = 200


def cost_ratio_grid(y_true, max_cost_ratio=DEFAULT_MAX_COST_RATIO, n_steps=N_COST_STEPS):
    """Return ``(ratios, Pc)`` for R sweeping ``[1, max_cost_ratio]`` on a log scale."""
    prevalence = float(np.mean(y_true))
    ratios = np.logspace(0.0, np.log10(max_cost_ratio), n_steps)
    pc = (prevalence * ratios) / (prevalence * ratios + (1.0 - prevalence))
    return ratios, pc


def operating_points(y_true, y_prob):
    """Every ``(FPR, FNR)`` pair reachable by moving the decision threshold.

    The two trivial corners are included: keep every pipe (FPR=0, FNR=1) and
    flag every pipe (FPR=1, FNR=0). The pairs are built here rather than from
    ``sklearn.metrics.det_curve`` because whether that function pads the curve
    with those corners changed between scikit-learn releases, which silently
    changes the AUCC.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.all() or not y_true.any():
        raise ValueError("both classes must be present in y_true")

    order = np.argsort(-y_prob, kind="stable")
    scores, labels = y_prob[order], y_true[order]
    # one operating point per distinct score: keep the last index of each run
    last_of_run = np.r_[np.flatnonzero(np.diff(scores)), scores.size - 1]

    true_positives = np.cumsum(labels)[last_of_run]
    false_positives = np.cumsum(~labels)[last_of_run]

    fpr = np.r_[0.0, false_positives / (~labels).sum()]
    fnr = np.r_[1.0, 1.0 - true_positives / labels.sum()]
    return fpr, fnr


def normalized_expected_cost(y_true, y_prob, pc):
    """``EC(Pc) = min_t [ Pc * FNR(t) + (1 - Pc) * FPR(t) ]``."""
    fpr, fnr = operating_points(y_true, y_prob)
    pc = np.asarray(pc, dtype=float).reshape(-1, 1)
    return np.min(pc * fnr + (1.0 - pc) * fpr, axis=1)


def area_under_cost_curve(y_true, y_prob, max_cost_ratio=DEFAULT_MAX_COST_RATIO, n_steps=N_COST_STEPS):
    """AUCC, the integral of ``EC(Pc)`` over the swept cost space."""
    _, pc = cost_ratio_grid(y_true, max_cost_ratio, n_steps)
    return float(np.trapezoid(normalized_expected_cost(y_true, y_prob, pc), pc))


def log_loss_score(y_true, y_prob):
    """Mean negative log likelihood of the predicted probabilities."""
    return float(log_loss(np.asarray(y_true), np.asarray(y_prob, dtype=float)))


def brier_score(y_true, y_prob):
    """Mean squared error of the predicted probabilities."""
    return float(brier_score_loss(np.asarray(y_true), np.asarray(y_prob, dtype=float)))
