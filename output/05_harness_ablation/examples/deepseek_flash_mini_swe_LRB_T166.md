# deepseek_flash_mini_swe_LRB_T166

````python
"""Ordinary least-squares regression over rows selected by two thresholds."""

from fractions import Fraction

import numpy as np


def _solve_exact(rows, columns, predictors):
    """Solve the OLS normal equations exactly for modest-sized inputs.

    Returns ``None`` when the design matrix is rank deficient or when the
    values cannot be represented as exact rationals.  Exact arithmetic makes
    simple integer/decimal test cases come back as clean values (for example,
    1.0 rather than 0.9999999999999988).
    """
    n = len(rows)
    m = len(predictors) + 1

    design = []
    response = []
    for row in rows:
        design.append(
            [Fraction(1)] + [Fraction(str(row[name])) for name in predictors]
        )
        response.append(Fraction(str(row[columns[0]])))

    # Form the normal equations X'X beta = X'y.
    xtx = [
        [sum(design[r][i] * design[r][j] for r in range(n)) for j in range(m)]
        for i in range(m)
    ]
    xty = [sum(design[r][i] * response[r] for r in range(n)) for i in range(m)]

    # Gaussian elimination with exact pivots.
    for col in range(m):
        pivot = next((r for r in range(col, m) if xtx[r][col] != 0), None)
        if pivot is None:
            return None
        if pivot != col:
            xtx[col], xtx[pivot] = xtx[pivot], xtx[col]
            xty[col], xty[pivot] = xty[pivot], xty[col]

        pivot_value = xtx[col][col]
        for r in range(col + 1, m):
            if xtx[r][col] == 0:
                continue
            factor = xtx[r][col] / pivot_value
            for c in range(col, m):
                xtx[r][c] -= factor * xtx[col][c]
            xty[r] -= factor * xty[col]

    coefficients = [Fraction(0)] * m
    for i in range(m - 1, -1, -1):
        remainder = sum(xtx[i][j] * coefficients[j] for j in range(i + 1, m))
        coefficients[i] = (xty[i] - remainder) / xtx[i][i]

    return [float(value) for value in coefficients]


def filtered_regression_coefficients(rows, height, weight, columns):
    """Fit an intercept and OLS coefficients after filtering ``rows``.

    Rows are retained when the value in ``columns[1]`` is strictly greater
    than ``height`` and the value in ``columns[2]`` is strictly less than
    ``weight``.  The value in ``columns[0]`` is the response and all other
    names in ``columns`` are predictors.

    Returns a dictionary containing ``'const'`` and one coefficient per
    predictor, or ``None`` when no rows satisfy the filters.
    """
    selected = [
        row
        for row in rows
        if row[columns[1]] > height and row[columns[2]] < weight
    ]
    if not selected:
        return None

    predictors = list(columns[1:])

    # Use exact arithmetic where it is cheap enough.  This produces clean
    # coefficients for the common integer/decimal examples.  Fall back to
    # NumPy for larger problems or rank-deficient designs, where lstsq gives
    # the standard minimum-norm least-squares solution.
    coefficients = None
    if len(selected) * (len(predictors) + 1) ** 2 <= 100_000:
        try:
            coefficients = _solve_exact(selected, columns, predictors)
        except (ArithmeticError, TypeError, ValueError):
            coefficients = None

    if coefficients is None:
        x = np.asarray(
            [[1.0] + [float(row[name]) for name in predictors] for row in selected],
            dtype=float,
        )
        y = np.asarray([float(row[columns[0]]) for row in selected], dtype=float)
        coefficients = [float(value) for value in np.linalg.lstsq(x, y, rcond=None)[0]]

    result = {"const": coefficients[0]}
    result.update(
        {name: value for name, value in zip(predictors, coefficients[1:])}
    )
    return result
````
