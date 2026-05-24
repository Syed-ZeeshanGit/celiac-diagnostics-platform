# src/model_engine.py
import numpy as np
from typing import Tuple

class VectorizedLogisticRegression:
    """
    An optimized, vectorized Logistic Regression engine implemented in pure NumPy.
    Executes cost function minimization via matrix gradient descent optimization.
    """
    def __init__(self, learning_rate: float = 0.1, epochs: int = 1000):
        self.lr: float = learning_rate
        self.epochs: int = epochs
        self.weights: np.ndarray = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> list:
        # Append a structural column of ones to handle intercept bias math seamlessly
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        self.weights = np.zeros(X_bias.shape[1])
        loss_history = []
        
        n_samples = X.shape[0]
        for _ in range(self.epochs):
            z = np.dot(X_bias, self.weights)
            predictions = self._sigmoid(z)
            
            # Compute current cost using Binary Cross-Entropy formulation
            cost = -np.mean(y * np.log(predictions + 1e-15) + (1.0 - y) * np.log(1.0 - predictions + 1e-15))
            loss_history.append(cost)
            
            # Calculate matrix gradients and shift parameter weights
            gradient = np.dot(X_bias.T, (predictions - y)) / n_samples
            self.weights -= self.lr * gradient
            
        return loss_history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        return self._sigmoid(np.dot(X_bias, self.weights))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class MedicalImageMatrixEngine:
    """
    Executes high-performance dimensional downscaling transformations on 2D matrices.
    Simulates feature map activations inside medical imaging networks.
    """
    @staticmethod
    def compute_max_pooling2d(matrix_channels: np.ndarray, stride_dimension: int = 4) -> np.ndarray:
        """
        Performs 2D Max Pooling from scratch over a grayscale diagnostic array.
        Time Complexity: O(N * M) structural cell evaluation.
        """
        h, w = matrix_channels.shape
        out_h = h // stride_dimension
        out_w = w // stride_dimension
        pooled_output = np.zeros((out_h, out_w))
        
        for r in range(out_h):
            for c in range(out_w):
                r_start = r * stride_dimension
                c_start = c * stride_dimension
                cell_block = matrix_channels[r_start:r_start+stride_dimension, c_start:c_start+stride_dimension]
                pooled_output[r, c] = np.max(cell_block)
                
        return pooled_output