import numpy as np

from training_lead_generation_model import optimize_threshold_fbeta, threshold_for_top_k, metrics_at_threshold


def test_threshold_for_top_k():
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    threshold = threshold_for_top_k(scores, 2)
    assert threshold == 0.8


def test_optimize_threshold_fbeta_bounds():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    threshold, fbeta = optimize_threshold_fbeta(y_true, y_score, beta=2.0)
    assert 0.0 <= threshold <= 1.0
    assert fbeta >= 0.0


def test_metrics_at_threshold_precision_recall():
    y_true = np.array([0, 1, 1, 0])
    y_score = np.array([0.2, 0.9, 0.8, 0.1])
    metrics = metrics_at_threshold(y_true, y_score, threshold=0.5, beta=2.0)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
