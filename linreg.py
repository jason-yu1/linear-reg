import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def normalize(X):  # creating standard variables here (u-x)/sigma
    if isinstance(X, pd.DataFrame):
        for c in X.columns:
            if is_numeric_dtype(X[c]):
                u = np.mean(X[c])
                s = np.std(X[c])
                X[c] = (X[c] - u) / s
        return X
    for j in range(X.shape[1]):
        u = np.mean(X[:, j])
        temp = X[:, j]
        s = np.std(X[:, j])
        X[:, j] = (X[:, j] - u) / s
    return X

def loss_gradient(X, y, B, lmbda):
    # X = np.c_[np.ones(len(X[:, 0])), X]
    return -X.T @ (y - X @ B)


def loss_ridge(X, y, B, lmbda):
    return (y - X @ B) @ (y - X @ B) + (lmbda * B @ B)


def loss_gradient_ridge(X, y, B, lmbda):
    return -X.T @ (y - X @ B) + (lmbda * B)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_likelihood(X, y, B, lmbda):
    return -np.sum(y @ (X @ B) - np.log(1 + np.exp(X @ B)))


def log_likelihood_gradient(X, y, B, lmbda):
    return -X.T @ (y - sigmoid(X @ B))


# NOT REQUIRED but to try to implement for fun
def L1_log_likelihood(X, y, B, lmbda):
    pass


# NOT REQUIRED but to try to implement for fun
def L1_log_likelihood_gradient(X, y, B, lmbda):
    """
    Must compute \beta_0 differently from \beta_i for i=1..p.
    \beta_0 is just the usual log-likelihood gradient
    # See https://aimotion.blogspot.com/2011/11/machine-learning-with-python-logistic.html
    # See https://stackoverflow.com/questions/38853370/matlab-regularized-logistic-regression-how-to-compute-gradient
    """
    pass


def minimize(X, y, loss_gradient,
             eta=0.00001, lmbda=0.0,
             max_iter=1000, addB0=True,
             precision=1e-9):
    "Here are various bits and pieces you might want"
    if X.ndim != 2:
        raise ValueError("X must be n x p for p features")

    n, p = X.shape

    if y.shape != (n, 1):
        raise ValueError(f"y must be n={n} x 1 not {y.shape}")

    X = normalize(X)

    if addB0:
        # add column of 1s to X
        X = np.c_[np.ones(len(X[:, 0])), X]

        B = np.random.random_sample(size=(p + 1, 1)) * 2 - 1  # make between [-1,1)
        h = np.zeros(shape=(p + 1, 1))
    else:
        B = np.random.random_sample(size=(p, 1)) * 2 - 1  # make between [-1,1)
        h = np.zeros(shape=(p, 1))

    eps = 1e-5  # prevent division by 0
    loss = loss_gradient(X, y, B, lmbda)
    while np.linalg.norm(loss) > precision:
        if max_iter == 0:
            break
        loss = loss_gradient(X, y, B, lmbda)
        h += np.tensordot(loss, loss)
        B = B - eta * (loss / (np.sqrt(h) + eps))
        max_iter -= 1

    # prev_B = B

    return B


class LinearRegression621:  # REQUIRED
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(X, y,
                          loss_gradient,
                          self.eta,
                          self.lmbda,
                          self.max_iter)



class LogisticRegression621:  # REQUIRED
    "Use the above class as a guide."
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict_proba(self, X):
        """
        Compute the probability that the target is 1. Basically do
        the usual linear regression and then pass through a sigmoid.
        """
        n = X.shape[0]
        B0 = np.ones(shape=(n,1))
        X = np.hstack([B0, X])

        return 1/(1 + np.exp(-(X @ self.B)))

    def predict(self, X):
        """
        Call self.predict_proba() to get probabilities then, for each x in X,
        return a 1 if P(y==1,x) > 0.5 else 0.
        """
        probs = self.predict_proba(X)

        return np.where(probs > 0.5, 1, 0)

    def fit(self, X, y):
        self.B = minimize(X, y,
                          log_likelihood_gradient,
                          self.eta,
                          self.lmbda,
                          self.max_iter)


class RidgeRegression621:  # REQUIRED
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(X, y,
                          loss_gradient_ridge,
                          self.eta,
                          self.lmbda,
                          self.max_iter,
                          addB0=False)
        self.B = np.append(np.mean(y), self.B)


# NOT REQUIRED but to try to implement for fun
class LassoLogistic621:
    pass
