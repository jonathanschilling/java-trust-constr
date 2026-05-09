#!/usr/bin/env python3
"""
One-shot generator for scipy reference values used in Java parity tests.

Run from the repository root:

    python3 src/test/python/regenerate_references.py

Each function below corresponds to a Java integration test under
`src/test/java/de/labathome/optimization/integration/`; the printed values are
copy-pasted into the test as the expected fixture. Re-run when scipy is
upgraded or a new test case is added — every reference value used by a Java
test should originate here so it can be reproduced.

Requires: scipy >= 1.10 (any version with `trust-constr` is fine).
"""

import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint


def _print_result(title, result):
    print(f"--- {title} ---")
    print(f"  x        = {np.array2string(np.asarray(result.x), precision=15, separator=', ')}")
    print(f"  fun      = {repr(float(result.fun))}")
    print(f"  nit      = {result.nit}")
    print(f"  status   = {result.status}")
    print(f"  message  = {result.message!r}")
    if hasattr(result, "optimality"):
        print(f"  opt      = {repr(float(result.optimality))}")
    if hasattr(result, "constr_violation"):
        print(f"  cv       = {repr(float(result.constr_violation))}")
    print()


def quadratic_on_hyperplane():
    """min x[0]^2 + x[1]^2  s.t.  x[0] + x[1] = 2.

    Closed form: optimum at (1, 1), f* = 2.
    """
    fun = lambda x: x[0] ** 2 + x[1] ** 2
    grad = lambda x: np.array([2 * x[0], 2 * x[1]])
    hess = lambda x: np.array([[2.0, 0.0], [0.0, 2.0]])
    A = np.array([[1.0, 1.0]])
    constraint = LinearConstraint(A, 2.0, 2.0)
    x0 = np.array([-1.0, 0.5])
    result = minimize(fun, x0, jac=grad, hess=hess, constraints=[constraint],
                      method="trust-constr", options={"xtol": 1e-10, "gtol": 1e-10})
    _print_result("quadratic_on_hyperplane", result)


def rosenbrock_on_hyperplane():
    """Rosenbrock with constraint x[0] + x[1] = 2.

    The unconstrained min (1, 1) lies on this hyperplane, so the constrained
    solution coincides with it.
    """
    fun = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    def grad(x):
        a, b = x
        return np.array([
            -2 * (1 - a) - 400 * a * (b - a ** 2),
            200 * (b - a ** 2),
        ])

    def hess(x):
        a, b = x
        return np.array([
            [2 - 400 * b + 1200 * a ** 2, -400 * a],
            [-400 * a, 200],
        ])
    A = np.array([[1.0, 1.0]])
    constraint = LinearConstraint(A, 2.0, 2.0)
    x0 = np.array([0.5, 1.5])
    result = minimize(fun, x0, jac=grad, hess=hess, constraints=[constraint],
                      method="trust-constr", options={"xtol": 1e-10, "gtol": 1e-10})
    _print_result("rosenbrock_on_hyperplane", result)


def quadratic_inside_disk():
    """min (x-2)^2 + (y-2)^2  s.t.  x^2 + y^2 <= 1.

    Optimum on disk boundary closest to (2, 2): (1/sqrt(2), 1/sqrt(2)).
    """
    fun = lambda x: (x[0] - 2) ** 2 + (x[1] - 2) ** 2
    grad = lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] - 2)])
    hess = lambda x: np.array([[2.0, 0.0], [0.0, 2.0]])
    constraint = NonlinearConstraint(
        lambda x: x[0] ** 2 + x[1] ** 2,
        -np.inf, 1.0,
        jac=lambda x: [[2 * x[0], 2 * x[1]]],
    )
    x0 = np.array([0.5, 0.5])
    result = minimize(fun, x0, jac=grad, hess=hess, constraints=[constraint],
                      method="trust-constr", options={"xtol": 1e-10, "gtol": 1e-10})
    _print_result("quadratic_inside_disk", result)


def hyperbolic_ineq():
    """Nocedal & Wright 15.1.

        minimize  (x[0] - 2)^2 / 2 + (x[1] - 1/2)^2 / 2
        s.t.      1/(x[0] + 1) - x[1] >= 1/4, x >= 0
    """
    fun = lambda x: 0.5 * (x[0] - 2) ** 2 + 0.5 * (x[1] - 0.5) ** 2
    grad = lambda x: np.array([x[0] - 2, x[1] - 0.5])
    hess = lambda x: np.eye(2)
    constraint = NonlinearConstraint(
        lambda x: 1.0 / (x[0] + 1) - x[1],
        0.25, np.inf,
        jac=lambda x: [[-1.0 / (x[0] + 1) ** 2, -1.0]],
    )
    from scipy.optimize import Bounds
    bounds = Bounds([0.0, 0.0], [np.inf, np.inf])
    x0 = np.array([0.5, 0.5])
    result = minimize(fun, x0, jac=grad, hess=hess,
                      constraints=[constraint], bounds=bounds,
                      method="trust-constr", options={"xtol": 1e-10, "gtol": 1e-10})
    _print_result("hyperbolic_ineq", result)


def eq_ineq_rosenbrock():
    """Rosenbrock with combined linear eq + ineq."""
    rosen = lambda x: 100.0 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    rosen_g = lambda x: np.array([
        -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
        200 * (x[1] - x[0] ** 2),
    ])
    rosen_h = lambda x: np.array([
        [2 - 400 * x[1] + 1200 * x[0] ** 2, -400 * x[0]],
        [-400 * x[0], 200],
    ])
    A_ineq = np.array([[1.0, 2.0]])
    A_eq = np.array([[2.0, 1.0]])
    constraints = [
        LinearConstraint(A_eq, 1.0, 1.0),
        LinearConstraint(A_ineq, -np.inf, 1.0),
    ]
    x0 = np.array([0.4, 0.2])
    result = minimize(rosen, x0, jac=rosen_g, hess=rosen_h,
                      constraints=constraints,
                      method="trust-constr", options={"xtol": 1e-10, "gtol": 1e-10})
    _print_result("eq_ineq_rosenbrock", result)


def main():
    print("# References for de/labathome/optimization/integration/*.java")
    print("# Generated by src/test/python/regenerate_references.py")
    print(f"# scipy = {__import__('scipy').__version__}")
    print()
    quadratic_on_hyperplane()
    rosenbrock_on_hyperplane()
    quadratic_inside_disk()
    hyperbolic_ineq()
    eq_ineq_rosenbrock()


if __name__ == "__main__":
    main()
