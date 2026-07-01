import pytest
import numpy as np
from src.models.train import asymmetric_financial_objective

def test_asymmetric_loss_direction_penalty():
    """Ensures that predicting the wrong directional sign results in heavier penalties."""
    # Scenario A: Correct direction, missed by 0.5
    y_true_a = np.array([0.5])
    y_pred_a = np.array([1.0])
    grad_a, _ = asymmetric_financial_objective(y_true_a, y_pred_a)
    
    # Scenario B: Wrong direction, missed by 0.5
    y_true_b = np.array([-0.25])
    y_pred_b = np.array([0.25])
    grad_b, _ = asymmetric_financial_objective(y_true_b, y_pred_b)
    
    # Absolute gradient of Scenario B must be strictly larger than Scenario A
    assert np.abs(grad_b[0]) > np.abs(grad_a[0]), "Directional errors must yield larger gradients."