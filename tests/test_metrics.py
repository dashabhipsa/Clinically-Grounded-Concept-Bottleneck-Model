"""Tests for multi-label classification metric calculations."""
import numpy as np

from src.evaluation.classification_metrics import (
    compute_binary_metrics,
    compute_multilabel_metrics,
    sigmoid,
)


def test_sigmoid_bounds():
    s = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert s[0] < 1e-6
    assert abs(s[1] - 0.5) < 1e-6
    assert s[2] > 1 - 1e-6


def test_auroc_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.9, 0.8])
    m = compute_binary_metrics(y_true, y_score)
    assert m["auroc"] == 1.0
    assert m["auprc"] == 1.0


def test_auroc_reversed_ranking():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.9, 0.8, 0.1, 0.2])
    m = compute_binary_metrics(y_true, y_score)
    assert abs(m["auroc"] - 0.0) < 1e-6


def test_sensitivity_specificity():
    # logits with large margins -> after sigmoid the 0.5 threshold yields
    # y_pred = [1, 1, 0, 0, 0, 1] (TP=2, FN=1, FP=1, TN=2)
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_score = np.array([5.0, 5.0, -5.0, -5.0, -5.0, 5.0])
    m = compute_binary_metrics(y_true, y_score, threshold=0.5)
    assert m["recall"] == 2 / 3
    assert m["sensitivity"] == m["recall"]
    assert m["specificity"] == 2 / 3
    assert m["precision"] == 2 / 3


def test_f1_known():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_score = np.array([5.0, 5.0, -5.0, -5.0, -5.0, 5.0])
    m = compute_binary_metrics(y_true, y_score, threshold=0.5)
    assert abs(m["f1"] - 2 / 3) < 1e-6


def test_single_class_present_returns_nan_roc():
    y_true = np.array([1, 1, 1, 1])
    y_score = np.array([0.9, 0.8, 0.7, 0.6])
    m = compute_binary_metrics(y_true, y_score)
    assert np.isnan(m["auroc"])
    assert np.isnan(m["auprc"])
    assert m["recall"] == 1.0  # still computable


def test_multilabel_macro_average():
    y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y_score = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.8], [0.2, 0.2]])
    result = compute_multilabel_metrics(y_true, y_score, class_names=["a", "b"])
    assert set(result["per_class"].keys()) == {"a", "b"}
    assert set(result["macro"].keys()) >= {
        "auroc", "auprc", "f1", "precision", "recall", "sensitivity", "specificity",
    }
    macro = result["macro"]
    expected = np.mean(
        [
            compute_binary_metrics(y_true[:, 0], y_score[:, 0])["auroc"],
            compute_binary_metrics(y_true[:, 1], y_score[:, 1])["auroc"],
        ]
    )
    assert abs(macro["auroc"] - expected) < 1e-6


def test_multilabel_with_empty_class_is_nan_safe():
    y_true = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])
    y_score = np.array([[0.9, 0.5], [0.1, 0.4], [0.8, 0.3], [0.2, 0.6]])
    result = compute_multilabel_metrics(y_true, y_score, class_names=["has_pos", "no_pos"])
    assert np.isnan(result["per_class"]["no_pos"]["auroc"])
    # macro must skip the empty class rather than propagate nan
    assert not np.isnan(result["macro"]["auroc"])
