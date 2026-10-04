from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GapDecomposition:
    observed_gap: float
    within_cohort_reference_term: float
    transfer_term: float


def decompose_gap(auc_dev_cv: float, auc_target_cv: float, auc_dev_to_target: float) -> GapDecomposition:
    """Manuscript Eq. (1).

    AUC_D^CV - AUC_D->R =
        (AUC_D^CV - AUC_R^CV) + (AUC_R^CV - AUC_D->R)
    """
    ref = auc_dev_cv - auc_target_cv
    transfer = auc_target_cv - auc_dev_to_target
    observed = auc_dev_cv - auc_dev_to_target
    return GapDecomposition(observed, ref, transfer)
