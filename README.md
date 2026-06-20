# Linear Models, Regularization, and Classification

This repository contains my implementation for AI-ENG-201 Homework 2: linear regression, regularization, logistic regression, Naive Bayes, and a simple text-classification pipeline.

## Project structure

```text
.
├── README.md
├── requirements.txt
├── pytest.ini
├── report.tex
├── report.pdf
├── src/
│   ├── linear\_regression.py          # Closed-form OLS
│   ├── linear\_regression\_gd.py       # Batch gradient-descent OLS
│   ├── ridge\_regression.py           # Closed-form Ridge regression
│   ├── lasso\_regression.py           # Coordinate-descent Lasso regression
│   ├── logistic\_regression.py        # Binary + OvR logistic regression
│   ├── naive\_bayes.py                # Gaussian Naive Bayes
│   ├── text\_features.py              # BagOfWords and TfidfTransformer
│   └── weighted\_linear\_regression.py # Bonus: Weighted least squares
├── notebooks/
│   └── hw2\_analysis.ipynb
├── tests/
│   └── test\_linear\_models.py
└── figures/
    ├── part1\_gd\_mse.pdf
    ├── part1\_polynomial\_overfitting.pdf
    ├── part2\_diabetes\_coefficient\_paths.pdf
    └── part3\_text\_roc.pdf
```

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

The assignment uses `fetch\_california\_housing` and `fetch\_20newsgroups`, so the first notebook run may need internet access or an existing local sklearn cache.

## Run the tests

```bash
pytest -q
```

Current status after adding the weighted-regression bonus test:

```text
9 passed
```

The tests compare the main implementations against sklearn equivalents where appropriate and also check shapes/numerical behavior for logistic regression, Naive Bayes, and the text feature transformers.

## Run the analysis notebook

```bash
jupyter notebook notebooks/hw2_analysis.ipynb
```

The notebook currently covers the main implemented experiments for OLS, gradient descent, polynomial fitting, Diabetes coefficient paths, Wine classification, and binary 20 Newsgroups text classification. A bonus cell for `WeightedLinearRegression` has been added at the end.

## Implemented models

### LinearRegression

`src/linear\_regression.py` implements closed-form ordinary least squares with an intercept by augmenting the design matrix with a column of ones.

### LinearRegressionGD

`src/linear\_regression\_gd.py` implements batch gradient descent on MSE. It stores the learned weights, coefficients, intercept, number of iterations, and `mse\_history\_`.

### RidgeRegression

`src/ridge\_regression.py` implements closed-form Ridge regression. The intercept term is not regularized.

### LassoRegression

`src/lasso\_regression.py` implements cyclic coordinate descent with soft-thresholding. Features are standardized internally, and predictions are transformed back to the original feature scale.

### LogisticRegression

`src/logistic\_regression.py` implements binary logistic regression and multiclass one-vs-rest classification. The sigmoid is implemented in a numerically stable branch form.

### GaussianNaiveBayes

`src/naive\_bayes.py` estimates class priors, feature means, and feature variances, then predicts in log-space for numerical stability.

### BagOfWords and TfidfTransformer

`src/text\_features.py` implements a simple top-frequency vocabulary and the assignment TF-IDF formula from scratch, without using `sklearn.feature\_extraction`.

### WeightedLinearRegression bonus

`src/weighted\_linear\_regression.py` implements weighted least squares for positive sample weights:

\[
w = (X^T V X)^{-1} X^T V y
]

where `V = diag(v)`. The implementation avoids explicitly constructing `V` and instead multiplies rows by the sample-weight vector.

