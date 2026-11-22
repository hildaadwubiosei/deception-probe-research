"""
Linear probes: one logistic-regression classifier per layer, each with its
own StandardScaler (later layers have much larger activation norms than
earlier ones, so per-layer scaling matters for a fair comparison).

Probes are trained only on the training set's normal/triggered activations.
Held-out examples and confound-control examples are reserved for evaluation
and are never used for probe fitting or layer selection.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def train_layer_probes(train_activations, train_labels, n_layers=7, max_iter=1000, C=1.0):
    """Returns (probes, scalers, train_aurocs) - each a dict keyed by layer."""
    probes, scalers, train_aurocs = {}, {}, {}

    for layer in range(n_layers):
        X_train = train_activations[layer]
        y_train = train_labels

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        probe = LogisticRegression(max_iter=max_iter, C=C)
        probe.fit(X_train_scaled, y_train)

        pred_proba = probe.predict_proba(X_train_scaled)[:, 1]
        train_aurocs[layer] = roc_auc_score(y_train, pred_proba)

        probes[layer] = probe
        scalers[layer] = scaler

    return probes, scalers, train_aurocs


def probe_predict_proba(layer, activations, probes, scalers):
    """Apply an already-fit layer's probe+scaler to new activations.
    Never refits the scaler or probe - this is how fixed-probe evaluation
    (e.g. the independent validation on new places) must always be done."""
    X_scaled = scalers[layer].transform(activations[layer])
    return probes[layer].predict_proba(X_scaled)[:, 1]
