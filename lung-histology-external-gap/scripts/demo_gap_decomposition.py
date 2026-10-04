from lung_external_gap.gap import decompose_gap

x = decompose_gap(
    auc_dev_cv=0.843,
    auc_target_cv=0.559,
    auc_dev_to_target=0.644,
)
print(x)
assert abs(x.observed_gap - 0.199) < 1e-12
assert abs(x.within_cohort_reference_term - 0.284) < 1e-12
assert abs(x.transfer_term - (-0.085)) < 1e-12
