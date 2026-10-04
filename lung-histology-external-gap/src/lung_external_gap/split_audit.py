from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _fold_predictions(X, y, patient_id, policy="patient", n_splits=5,
                      pca_dim=64, classifier="knn", seed=2026):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    patient_id = np.asarray(patient_id)

    if policy == "slice":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(X, y)
    elif policy == "patient":
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(X, y, groups=patient_id)
    else:
        raise ValueError("policy must be 'slice' or 'patient'")

    pred = np.empty_like(y)
    prob = np.full(len(y), np.nan, dtype=float)
    same_patient_nn = np.full(len(y), False, dtype=bool)

    for tr, te in splits:
        scaler = StandardScaler(with_std=False).fit(X[tr])
        Xt = scaler.transform(X[tr])
        Xq = scaler.transform(X[te])

        pca = PCA(n_components=min(pca_dim, len(tr)-1, X.shape[1]), svd_solver="full")
        Zt = pca.fit_transform(Xt)
        Zq = pca.transform(Xq)

        if classifier == "knn":
            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(Zt, y[tr])
            pred[te] = clf.predict(Zq)
            if np.unique(y).size == 2:
                # 1-NN class score for AUC.
                cls = list(clf.classes_)
                pos = cls.index(sorted(cls)[-1])
                prob[te] = clf.predict_proba(Zq)[:, pos]

            # Direct memorisation audit.
            # Squared Euclidean distance is enough for argmin.
            aa = np.sum(Zq * Zq, axis=1, keepdims=True)
            bb = np.sum(Zt * Zt, axis=1, keepdims=True).T
            D2 = aa + bb - 2 * (Zq @ Zt.T)
            nn_local = np.argmin(D2, axis=1)
            same_patient_nn[te] = patient_id[tr][nn_local] == patient_id[te]

        elif classifier == "lr":
            clf = LogisticRegression(
                C=0.1, class_weight="balanced", max_iter=5000
            )
            clf.fit(Zt, y[tr])
            pred[te] = clf.predict(Zq)
            if np.unique(y).size == 2:
                prob[te] = clf.predict_proba(Zq)[:, 1]
        else:
            raise ValueError("classifier must be 'knn' or 'lr'")

    return pred, prob, same_patient_nn


def audit_split_policy(X, y, patient_id, classifier="knn",
                       n_splits=5, pca_dim=64, seed=2026):
    out = {}
    for policy in ["slice", "patient"]:
        pred, prob, same = _fold_predictions(
            X, y, patient_id, policy=policy, n_splits=n_splits,
            pca_dim=pca_dim, classifier=classifier, seed=seed
        )
        row = {
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        }
        if np.unique(y).size == 2 and np.all(np.isfinite(prob)):
            row["auc"] = float(roc_auc_score(y, prob))
        if classifier == "knn" and policy == "slice":
            row["nearest_training_slice_same_patient"] = float(np.mean(same))
        out[policy] = row
    return out
