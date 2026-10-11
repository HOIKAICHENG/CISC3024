import numpy as np

from code.metrics import classification_metrics, ood_metrics, predictive_entropy


def test_perfect_binary_predictions():
    probs = np.array([[0.9, 0.1], [0.1, 0.9]])

    result = classification_metrics(probs, np.array([0, 1]), bins=2)

    assert result["accuracy"] == 1.0
    assert np.isclose(result["nll"], -np.log(0.9))
    assert np.isclose(result["brier"], 0.02)


def test_ece_assigns_boundary_confidence_to_lower_bin():
    probs = np.array([[0.5, 0.5], [0.75, 0.25]])

    result = classification_metrics(probs, np.array([1, 0]), bins=2)

    assert np.isclose(result["ece"], 0.375)


def test_zero_and_one_probabilities_remain_finite():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])

    result = classification_metrics(probs, np.array([0, 1]), bins=2)
    entropy = predictive_entropy(probs)

    assert all(np.isfinite(value) for value in result.values())
    assert np.all(np.isfinite(entropy))
    assert np.allclose(entropy, 0.0, atol=1e-10)


def test_separated_ood_scores_are_perfect():
    result = ood_metrics(np.array([0.1, 0.2]), np.array([0.8, 0.9]))

    assert result["auroc"] == 1.0
    assert result["aupr"] == 1.0
    assert result["fpr95"] == 0.0


def test_uniform_prediction_has_more_entropy():
    assert predictive_entropy(np.array([[0.5, 0.5]]))[0] > predictive_entropy(
        np.array([[0.99, 0.01]])
    )[0]
