from typing import Tuple
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

def asymmetric_financial_objective(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Custom objective function for LightGBM.
    Penalizes directional errors (wrong sign) significantly harder than magnitude errors.
    
    Mathematical Formulation:
    Let e = y_pred - y_true
    Loss(e) = 0.5 * (e)^2  if sign(y_true) == sign(y_pred)
    Loss(e) = 2.0 * (e)^2  if sign(y_true) != sign(y_pred)
    """
    residual = y_pred - y_true
    
    # Check if signs match (True if both positive or both negative)
    same_sign = (y_true * y_pred) >= 0
    
    # Establish penalty multipliers
    multiplier = np.where(same_sign, 1.0, 4.0)
    
    # Calculate First Derivative (Gradient) and Second Derivative (Hessian)
    gradient = multiplier * residual
    hessian = multiplier
    
    return gradient, hessian

class AlphaPredictor:
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.05):
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            verbosity=-1,
            random_state=42
        )
        # Inject our custom financial loss function
        self.model.set_params(objective=asymmetric_financial_objective)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains the custom LightGBM model."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates real-time alpha predictions."""
        return self.model.predict(X)

if __name__ == "__main__":
    print("Generating simulated factor matrices for testing...")
    # Synthetic data structure matching our Polars factor dataframe
    # Features: [tax_burden, asset_turnover, leverage, nlp_negative_sentiment]
    np.random.seed(42)
    X_sample = np.random.normal(loc=0.0, scale=1.0, size=(1000, 4))
    
    # Target: 3-month forward excess return (Alpha)
    y_sample = 0.3 * X_sample[:, 1] - 0.5 * X_sample[:, 3] + np.random.normal(0, 0.1, size=1000)

    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2)

    predictor = AlphaPredictor()
    print("Training LightGBM model with Asymmetric Financial Loss...")
    predictor.train(X_train, y_train)
    
    predictions = predictor.predict(X_test)
    print(f"Model trained successfully. Sample predictions generated: {predictions[:5]}")