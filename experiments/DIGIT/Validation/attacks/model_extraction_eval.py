"""Comparative model extraction / surrogate training attack."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from ..comparative.querying import QueryAccounting, primitive_vector_from_system
from ..data.base import PrivateDataset
from ..data.query_gen import QueryGenerator


def run_model_extraction_attack(
    system: Any,
    private_data: PrivateDataset,
    seed: int,
    num_queries: int = 10000,
) -> Dict[str, float]:
    gen = QueryGenerator(private_data, include_specificity_field=True)
    queries = gen.generate(num_queries, seed=seed)
    accounting = QueryAccounting()
    epsilon = float(getattr(system, "epsilon", 0.0)) if hasattr(system, "epsilon") else None
    X: List[np.ndarray] = []
    y_answer: List[int] = []
    y_support: List[int] = []
    y_conf: List[int] = []
    y_risk: List[int] = []

    for q in queries:
        response = primitive_vector_from_system(system, q, private_data)
        accounting.record(1, epsilon)
        X.append(q.cpu().numpy().astype(float))
        y_answer.append(int(response["answer"]))
        y_support.append(int(response["support"]))
        y_conf.append(int(response["confidence"]))
        y_risk.append(int(response["risk"]))

    X_np = np.asarray(X, dtype=float)
    y_stack = {
        "answer": np.asarray(y_answer, dtype=int),
        "support": np.asarray(y_support, dtype=int),
        "confidence": np.asarray(y_conf, dtype=int),
        "risk": np.asarray(y_risk, dtype=int),
    }

    accuracies = {}
    for name, y in y_stack.items():
        X_tr, X_te, y_tr, y_te = train_test_split(X_np, y, test_size=0.2, random_state=seed, stratify=y)
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=seed)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        accuracies[f"{name}_accuracy"] = float(accuracy_score(y_te, pred))

    mean_acc = float(np.mean(list(accuracies.values())))
    return {
        "surrogate_accuracy": mean_acc,
        "query_count": accounting.query_count,
        "cumulative_epsilon": accounting.cumulative_epsilon,
        **accuracies,
    }
