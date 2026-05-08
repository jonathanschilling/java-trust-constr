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

## Known gaps (work-in-progress)

- **`MinimizeTrustConstr.minimizeTrustConstr` body** is incomplete past line ~329: canonical-form constraint concatenation, Lagrangian Hessian assembly, method dispatch (`TrustRegionInteriorPoint` vs `EqualityConstrainedSQP`), and the iteration/callback loop are still TODO. Inner methods (`TrustRegionInteriorPoint`, `EqualityConstrainedSQP`, `BarrierSubproblem`) are runnable; the orchestrator that wires them up is not.
- **`LinearConstraint`/`NonlinearConstraint`** support pure-equality and one-sided inequality only. Two-sided interval constraints (`lb < ub`, both finite) throw `UnsupportedOperationException` and need scipy's canonical-form row-splitting (`canonical_constraint.py`) ported.
- **Scipy test-suite parity**: `test_canonical_constraint.py`, `test_projections.py`, `test_qp_subproblem.py`, `test_nested_minimize.py`, and selected cases from `test_minimize_constrained.py` are not yet translated.
