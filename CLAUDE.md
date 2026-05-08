# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Java port of `scipy.optimize.minimize(method='trust-constr')` — a trust-region constrained optimizer. The reference Python implementation is checked out as a git submodule at `scipy/` (upstream `scipy/scipy`); the relevant sources live in `scipy/scipy/optimize/_trustregion_constr/`. When porting or debugging, compare each Java class against its Python counterpart of the same name (e.g. `Projections.java` ↔ `projections.py`, `QPSubproblem.java` ↔ `qp_subproblem.py`, `MinimizeTrustConstr.java` ↔ `minimize_trustregion_constr.py`).

To populate the submodule on a fresh clone: `git submodule update --init`.

## Build & test

Maven project, Java 11. Inherits from parent POM `de.labathome:de-labathome-parent` (must be installed locally — this is not a public artifact).

```
mvn compile             # build
mvn test                # run all tests
mvn -Dtest=TestNumDiff test                            # single test class
mvn -Dtest=TestNumDiff#testGroupColumns test           # single test method
```

Tests use JUnit 5 plus `MinervaAssertions` from an internal `minerva.tests.junit` package — another non-public dependency. If `mvn` fails resolving artifacts, the parent POM and Minerva are not on Maven Central; they need to be available in a local/internal repo.

## Code layout

- `src/main/java/org/scipy/optimize/minimize/` — the port itself. The package name is deliberately `org.scipy.*` to mirror upstream.
  - Top level: one class per scipy `.py` file (`Projections`, `QPSubproblem`, `EqualityConstrainedSQP`, `TrustRegionInteriorPoint`, `BarrierSubproblem`, `NumDiff`, `ScalarFunction`, `VectorFunction`, `BFGS`, `SR1`, `LinearConstraint`, `NonlinearConstraint`, `MinimizeTrustConstr`).
  - `interfaces/` — functional interfaces (`LinearOperator`, `Constraint`, `Jacobian`, `HessianProduct`, `HessianUpdateStrategy`, `StoppingCriterion`, …) that stand in for Python's duck typing.
  - `records/` — plain data carriers (`State`, `StateIP`, `Bounds`, `OptimizeResult`, `CGInfo`, `PreparedConstraint`, …). These correspond to scipy's namedtuples / ad-hoc dicts.
  - `enums/` — typed replacements for scipy's string-flag arguments (`ProjectionMethod`, `FiniteDifferenceMethod`, `HessianApproximationType`, `TrustConstrMethod`, …).
- `src/test/java/de/labathome/optimization/` — tests live in a different package (`de.labathome.optimization`) than the code under test. Each `Test<X>.java` targets one main-package class.

## Matrix library and the in-tree sparse module

The legacy linear algebra goes through **UJMP** (`org.ujmp.core.Matrix`). For BLAS/LAPACK the project uses **`dev.ludovic.netlib`** (luhenry/netlib, 3.2.0). The previous `com.github.fommil.netlib` dependency has been retired.

UJMP is being incrementally retired in favour of an in-tree sparse module at **`org.scipy.optimize.minimize.sparse`** (a faithful Java port of the slice of `scipy.sparse` that trust-constr actually uses). The previously-mentioned ojAlgo migration is **no longer planned** — the sparse module is the path forward. Key pieces:

- `CSRMatrix`, `CSCMatrix` — primitive-array (`int[] indptr/indices`, `double[] data`) sparse storage matching scipy's `csr_array`/`csc_array`.
- `SparseAssembly` — `vstack`, `hstack`, `blockArray` (the `[[A,B],[C,D]]` shape from `projections.py:99`), and `assembleJacobianWithSlacks` (the optimised KKT-Jacobian build from `tr_interior_point.py:_assemble_sparse_jacobian`).
- `DenseSolve` — LAPACK `dgetrf`/`dgetrs` wrapper for the KKT solve. Currently materialises the assembled CSR/CSC matrix to dense before factoring; replacing this with a true sparse LU is a single-call-site swap.
- `SparseLinearOperator` — adapts a CSR/CSC matvec to the existing `interfaces.LinearOperator`.
- `UjmpBridge` — UJMP↔CSR/CSC conversions used at the boundary of classes still typed in `Matrix`.

`LinAlg.java` is still temporary, but its retirement is tied to UJMP retirement, not an ojAlgo migration. Three trust-constr KKT paths have already been moved off UJMP onto the new module: `Projections.augmentedSystemProjections`, `QPSubproblem.eqpKktFact`, `BarrierSubproblem.computeJacobian`. These are the templates for the rest of the migration.

## Porting conventions

- Methods that mutate a `State`/`StateIP` in scipy are translated as static methods on `MinimizeTrustConstr` that take the state as a parameter and return it (see `updateState`, `updateStateIP`).
- Keep Javadoc references to the scipy paper citations and equation numbers — they're load-bearing for cross-referencing the algorithm.
- The development cadence (see `git log`) is "port one scipy function, write a parity test, commit." Match that — don't bulk-port without tests.

## Public entry points

- `MinimizeTrustConstr.minimizeEqualityConstrained(fun, grad, hess, x0, eq, maxIter, xtol, gtol)` — focused entry point covering the equality-constrained slice (no constraints, or a single `LinearConstraint` / `NonlinearConstraint` whose rows are all equalities). Dispatches to `EqualityConstrainedSQP.eqSQP` directly. Used by `src/test/java/de/labathome/optimization/integration/`.
- `MinimizeTrustConstr.minimizeTrustConstr(...)` — the full scipy-shaped API. **Body still incomplete past the prepared-constraint step**: canonical-form concatenation, Lagrangian Hessian assembly, and the inequality-method dispatch are TODO.

## Known gaps (work-in-progress)

- **`MinimizeTrustConstr.minimizeTrustConstr` body** is incomplete past line ~329: canonical-form constraint concatenation, Lagrangian Hessian assembly, method dispatch (`TrustRegionInteriorPoint` vs `EqualityConstrainedSQP`), and the iteration/callback loop are still TODO. Inner methods (`TrustRegionInteriorPoint`, `EqualityConstrainedSQP`, `BarrierSubproblem`) are runnable; the full orchestrator that wires them up is not. The narrower `minimizeEqualityConstrained` covers the equality-constrained subset.
- **`LinearConstraint`/`NonlinearConstraint`** support pure-equality and one-sided inequality only. Two-sided interval constraints (`lb < ub`, both finite) throw `UnsupportedOperationException` and need scipy's canonical-form row-splitting (`canonical_constraint.py`) ported.
- **Inequality-constrained optimization** (which goes through `TrustRegionInteriorPoint` + `BarrierSubproblem`) is not yet wired up by an orchestrator entry point. The pieces are in place; the final dispatch is not.
- **Scipy test-suite parity**: `test_canonical_constraint.py`, `test_projections.py`, `test_qp_subproblem.py`, `test_nested_minimize.py`, and selected cases from `test_minimize_constrained.py` are not yet translated. New scipy-reference values for Java integration tests live in `src/test/python/regenerate_references.py`.

## Bug fixes that landed alongside the orchestrator

A few latent bugs surfaced when wiring `EqualityConstrainedSQP` into a runnable orchestrator and were fixed:

- `EqualityConstrainedSQP.defaultScaling(n)` returned the input vector instead of the identity (the comment said "no scaling" but the body returned `x`). Now returns `Matrix.Factory.eye(n, n)`.
- `EqualityConstrainedSQP.eqSQP(...)` defaulted `trustUb` to `NEGATIVE_INFINITY` (typo). Now `POSITIVE_INFINITY`.
- `QPSubproblem.projectedCG`'s `reinforceBoxBoundaries` path went through `Matrix.Factory.importFromArray(double[])`, which produced a 1×n row matrix and silently broke shape consistency for downstream `dn + dt`. Now uses `UjmpBridge.arrayToCol(...)` which preserves the n×1 column convention. Four `TestProjectedCG` assertions had been written against the buggy row-shape; they were updated to column indexing.
- `ScalarFunction.ScalarFunctionFactory` constructor was private with a single shared `FACTORY` static instance whose state leaked between calls. Constructor is now `public` so callers can build a fresh factory per `minimize` invocation.
