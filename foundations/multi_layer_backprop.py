import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        x, W1, b1 = np.array(x), np.array(W1), np.array(b1)
        W2, b2, y = np.array(W2), np.array(b2), np.array(y_true)
        n = len(y)


        # forward
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)
        z2 = W2 @ a1 + b2
        loss = np.mean((z2 - y) ** 2)

        # backward
        dz2 = 2 * (z2 - y) / n
        dW2 = np.outer(dz2, a1)
        db2 = dz2
        da1 = W2.T @ dz2
        dz1 = da1 * (z1 > 0)        # masque binaire
        dW1 = np.outer(dz1, x)
        db1 = dz1

        r = lambda a: np.round(a, 4).tolist()
        return {'loss': round(float(loss), 4),
                'dW1': r(dW1), 'db1': r(db1),
                'dW2': r(dW2), 'db2': r(db2)}




