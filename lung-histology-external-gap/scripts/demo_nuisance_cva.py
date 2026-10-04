import numpy as np
from lung_external_gap.nuisance_cva import NuisanceCVA

rng = np.random.default_rng(7)
n = 240
y = np.repeat([0, 1], n//2)
g = np.tile(np.repeat([0,1,2,3], n//8), 2)

X = rng.normal(size=(n, 12))
X[:, 0] += 1.25 * y
X[:, 1] += 1.0 * g

m = NuisanceCVA(n_min=3, mode="hard").fit(X, y, g)
Z = m.transform(X)

print("nuisance rank:", m.V_.shape[1])
print("input shape:", X.shape, "output shape:", Z.shape)
