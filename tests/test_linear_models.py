import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.linear_model import Ridge as SklearnRidge

from src.linear_regression import LinearRegression
from src.linear_regression_gd import LinearRegressionGD
from src.ridge_regression import RidgeRegression
from src.lasso_regression import LassoRegression

def test_linear_regression_matches_sklearn_predictions():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 5))
    y = rng.normal(size=120)
    X_test = rng.normal(size=(20, 5))

    custom = LinearRegression().fit(X, y)
    sklearn = SklearnLinearRegression().fit(X, y)

    assert np.allclose(custom.predict(X_test), sklearn.predict(X_test), atol=1e-9)


def test_linear_regression_gd_gets_close_to_closed_form():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(250, 4))
    true_w = np.array([2.0, -1.0, 0.5, 3.0])
    y = 1.5 + X @ true_w + rng.normal(scale=0.05, size=250)

    closed_form = LinearRegression().fit(X, y)
    gd = LinearRegressionGD(lr=0.05, max_iter=20000, tol=1e-8).fit(X, y)

    assert np.allclose(gd.weights_, closed_form.weights_, atol=1e-4)
    assert len(gd.mse_history_) == gd.n_iter_


def test_ridge_regression_matches_sklearn_predictions():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(150, 6))
    y = rng.normal(size=150)
    X_test = rng.normal(size=(15, 6))

    lambda_ = 2.5
    custom = RidgeRegression(lambda_=lambda_).fit(X, y)
    sklearn = SklearnRidge(alpha=lambda_, fit_intercept=True, solver="cholesky").fit(X, y)

    assert np.allclose(custom.predict(X_test), sklearn.predict(X_test), atol=1e-9)


def test_lasso_regression_shrinks_some_coefficients():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 5))
    y = 2.0 * X[:, 0] - 3.0 * X[:, 2] + rng.normal(scale=0.1, size=200)

    model = LassoRegression(lambda_=10.0, max_iter=10000, tol=1e-5).fit(X, y)

    assert model.coef_ is not None
    assert np.count_nonzero(np.abs(model.coef_) < 1e-2) >= 1
    assert model.predict(X[:5]).shape == (5,)