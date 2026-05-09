# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

(no changes yet)

## [1.0.0] - 2026-05-09

Initial release of the Java port of
`scipy.optimize.minimize(method='trust-constr')`.

### Added

- **Two solver paths**, with automatic dispatch by constraint shape:
  - `EqualityConstrainedSQP`: Byrd-Omojokun trust-region SQP for
    equality-only and unconstrained problems. Iteration-count parity with
    scipy on the equality-constrained quadratic (3 iterations) and
    Rosenbrock (7 iterations).
  - `TrustRegionInteriorPoint` + `BarrierSubproblem`: log-barrier interior
    point for any problem with inequality constraints (or bounds).
- **Constraint types**:
  - `LinearConstraint` and `NonlinearConstraint` with two-sided
    `lb <= f(x) <= ub` bounds, per-row `keepFeasible` enforcement, the
    scipy 4-block canonical-row ordering (less / greater / interval-upper /
    interval-lower), and an analytic constraint Hessian-of-Lagrangian.
  - `Bounds` (folded into the constraint set as a `LinearConstraint` with
    identity `A`).
  - `CombinedConstraint` for multi-source concatenation; `CanonicalConstraint`
    for the scipy-shape `f_eq=0, f_ineq<=0` API including
    `concatenate`/`empty`/`fromPreparedConstraint` factories and
    `initialConstraintsAsCanonical`.
- **Quasi-Newton fallbacks**: `BFGS` and `SR1` Hessian-update strategies
  (both `HESSIAN` and `INV_HESSIAN` modes), plus 2-point/3-point
  finite-difference gradient approximation when no analytic gradient is
  supplied. Mirrors scipy's default behaviour when `hess` / `grad` are
  omitted.
- **Sparse-aware** Jacobians: `LinearConstraint` and `NonlinearConstraint`
  preserve sparsity end-to-end; `Projections` auto-routes sparse problems
  through the AUGMENTED_SYSTEM factorization. In-tree CSR/CSC sparse
  module (`CSRMatrix`, `CSCMatrix`, `SparseAssembly`, `DenseSolve`).
- **Public entry points** on `MinimizeTrustConstr`:
  - 5 convenience overloads for `minimize(...)`, ranging from "objective
    only" (FD grad, BFGS hess) up to fully-analytic.
  - `minimizeTrustConstr(...)`: full scipy-shape API with `args`, `bounds`,
    `hessp`, `callback`, `verbose`/`disp` per-iteration progress
    (`SQPReport` / `IPReport`), `factorizationMethod`, four `initial*`
    tuning knobs, `sparseJacobian`, `finiteDifferenceRelStep`, and
    `workers` for parallel FD evaluation.
- **NumDiff**: `approxDerivative`, `approxDerivativeAsLinearOperator`,
  `approxDerivativeWithWorkers` (parallel column FD via `ExecutorService`),
  `approxDerivativeFullOutput`, plus Curtis-Powell-Reid column grouping
  for sparse-pattern Jacobians.
- **Matrix layer**: column-major dense storage with zero-copy LAPACK/BLAS
  calls (`dgesv` / `dgeqrf`+`dorgqr` / `dgesvd` / `dpotrf` / `dgemm` /
  `dsyr` / `dsyr2`) via `dev.ludovic.netlib`. Static facade `LinAlg` with
  factorization-result records (`QRResult`, `SVDResult`, `CholResult`).
- **283 tests**, with strict 1:1 parity for scipy's
  `_trustregion_constr/tests/test_projections.py` (10/10),
  `test_qp_subproblem.py` (23/23),
  `test_canonical_constraint.py` (6/6), `test_report.py` (2/2), and
  `test_nested_minimize.py` (1/1). Selected ports of
  `test_minimize_constrained.py`, `test_hessian_update_strategy.py`,
  `test_constraints.py`, `test_differentiable_functions.py`,
  `test__numdiff.py`. Parametrized `TestListOfProblems` over Rosenbrock,
  IneqRosenbrock, Maratos, HyperbolicIneq with analytic + FD-gradient
  axes. `TestRosenbrockNoException` over scipy's hand-encoded 38-point
  Rosenbrock iteration sequence for BFGS / SR1.
- **Javadoc**: 0 errors, 0 warnings on Java 17 (`maven-javadoc-plugin`
  3.11.2). Published to GitHub Pages on every push to `master`
  (<https://jonathanschilling.github.io/java-trust-constr/>).

### Out of scope

The following scipy features are intentionally not ported in v1.0.0; each
is a meaningful chunk of work, not a bug.

- **SuperLU sparse direct factorization.** The KKT solve in
  `QPSubproblem.eqpKktFact` materializes the assembled CSR/CSC matrix to
  dense via `DenseSolve` (LAPACK `dgetrf`/`dgetrs`). Replacing this with a
  true sparse LU is a single-call-site swap when a Java sparse-LU library
  is integrated.
- **Complex-step (`'cs'`) finite differences.** `FiniteDifferenceMethod.COMPLEX_STEP`
  exists as an enum value but the implementation throws "not implemented
  yet". Less commonly used; scipy's default 2-point/3-point methods are
  fully supported.
- **`'NormalEquation'` projection factorization for sparse Jacobians.**
  Requires Cholesky of `A A^T` via scikit-sparse, which has no Java
  equivalent that ships with the netlib BLAS/LAPACK we already depend on.
  `'AugmentedSystem'` (the scipy default for sparse) is supported.

### Notes

- Built on Java 17, parent POM `de.labathome:de-labathome-parent:1.1.0`,
  runtime dependencies `dev.ludovic.netlib:blas/lapack:3.2.0` (both
  publicly available on Maven Central).
- Sources are ASCII-only as a hard repo invariant.

[Unreleased]: https://github.com/jonathanschilling/java-trust-constr/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/jonathanschilling/java-trust-constr/releases/tag/v1.0.0
