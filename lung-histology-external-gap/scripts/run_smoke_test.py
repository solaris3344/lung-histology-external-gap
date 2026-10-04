import numpy as np

from lung_external_gap.gap import decompose_gap
from lung_external_gap.nuisance_cva import NuisanceCVA, class_retention_ratio
from lung_external_gap.combat import LocationScaleComBat
from lung_external_gap.bootstrap import bootstrap_auc
from lung_external_gap.size_analysis import tertile_auc

g = decompose_gap(0.843, 0.559, 0.644)
assert round(g.observed_gap, 3) == 0.199
assert round(g.within_cohort_reference_term, 3) == 0.284
assert round(g.transfer_term, 3) == -0.085

rng = np.random.default_rng(2026)
n = 160
y = np.repeat([0,1], n//2)
groups = np.tile(np.repeat([0,1,2,3], n//8), 2)
X = rng.normal(size=(n, 16))
X[:,0] += y
X[:,1] += 0.8 * groups

cva = NuisanceCVA(n_min=3, mode="hard").fit(X, y, groups)
Z = cva.transform(X)
assert Z.shape == X.shape
rho = class_retention_ratio(cva.P_, X[y==1].mean(0), X[y==0].mean(0))
assert np.isfinite(rho)

combat = LocationScaleComBat().fit(X, y, groups)
Xh = combat.transform_training(X, y, groups)
assert Xh.shape == X.shape

score = X[:,0]
auc, lo, hi = bootstrap_auc(y, score, n_boot=100, seed=1)
assert 0 <= auc <= 1 and 0 <= lo <= 1 and 0 <= hi <= 1

diam = np.exp(rng.normal(size=n))
rows = tertile_auc(score, diam, y)
assert len(rows) == 3

print("Smoke test: OK")
