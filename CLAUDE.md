# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## HARD RULE: ASCII-only

**Every file in this repo MUST contain only ASCII characters (bytes 0x00-0x7F).** No exceptions, no `--`, no smart quotes, no Unicode arrows / Greek letters / em-dashes / non-breaking spaces -- no matter the file type. This applies to source code, tests, build files (`pom.xml`), documentation (`README.md`, `CLAUDE.md`), Python helpers (`src/test/python/*.py`), config files, and anything else committed to the repo. The `scipy/` submodule is exempt because it is checked out from upstream as-is.

When writing or editing any file, use ASCII substitutions (e.g. `->` for `->`, `--` for `--`, `^2` for superscript-2, `lambda` for the Greek letter, `>=` / `<=` / `!=` / `~=` for the typographic comparison glyphs, `'` and `"` for straight quotes only). The existing source tree was scrubbed once with a deliberate pass; do not regress.

To audit before committing:

```bash
LC_ALL=C python3 -c "
import os
for dirpath, dirs, files in os.walk('.'):
    for skip in ['.git', 'target', 'scipy', 'node_modules']: dirs[:] = [d for d in dirs if d != skip]
    for fn in files:
        path = os.path.join(dirpath, fn)
        with open(path, 'rb') as f: data = f.read()
        if any(b > 127 for b in data): print(path)
"
```

(GNU `grep -rlP "[\x80-\xff]"` does NOT reliably catch UTF-8 bytes in modern locales -- the Python scan above is the reliable check.)

## Project purpose

Java port of `scipy.optimize.minimize(method='trust-constr')` -- a trust-region constrained optimizer. The reference Python implementation is checked out as a git submodule at `scipy/` (upstream `scipy/scipy`); the relevant sources live in `scipy/scipy/optimize/_trustregion_constr/`. When porting or debugging, compare each Java class against its Python counterpart of the same name (e.g. `Projections.java` <-> `projections.py`, `QPSubproblem.java` <-> `qp_subproblem.py`, `MinimizeTrustConstr.java` <-> `minimize_trustregion_constr.py`).

To populate the submodule on a fresh clone: `git submodule update --init`.

## Build & test

Maven project, Java 17. The child POM overrides the parent's Java 1.8 default via `maven-compiler-plugin` 3.11.0 with `<release>17</release>`, and pins surefire to 3.2.5 so the test forks work on a 17 JRE. Inherits from parent POM `de.labathome:de-labathome-parent` (must be installed locally -- this is not a public artifact).

```
mvn compile             # build
mvn test                # run all tests
mvn -Dtest=TestNumDiff test                            # single test class
mvn -Dtest=TestNumDiff#testGroupColumns test           # single test method
```

If your default JDK is older than 17, set `JAVA_HOME` for the build, e.g. `JAVA_HOME=/usr/lib/jvm/java-17-openjdk mvn test`.

Tests use JUnit 5 plus an in-tree `de.labathome.optimization.RelAbsAssertions` helper for rel/abs floating-point comparisons. If `mvn` fails resolving artifacts, the parent POM is not on Maven Central; it needs to be available in a local/internal repo.

## Code layout

- `src/main/java/de/labathome/trustconstr/` -- the port itself. Originally lived under `org.scipy.optimize.minimize` (mirroring upstream); migrated to `de.labathome.trustconstr` for Maven Central publishing under the owner's verified namespace.
  - Top level: one class per scipy `.py` file (`Projections`, `QPSubproblem`, `EqualityConstrainedSQP`, `TrustRegionInteriorPoint`, `BarrierSubproblem`, `NumDiff`, `ScalarFunction`, `VectorFunction`, `BFGS`, `SR1`, `LinearConstraint`, `NonlinearConstraint`, `MinimizeTrustConstr`).
  - `interfaces/` -- functional interfaces (`LinearOperator`, `Constraint`, `Jacobian`, `HessianProduct`, `HessianUpdateStrategy`, `StoppingCriterion`, ...) that stand in for Python's duck typing.
  - `records/` -- plain data carriers (`State`, `StateIP`, `Bounds`, `OptimizeResult`, `CGInfo`, `PreparedConstraint`, ...). These correspond to scipy's namedtuples / ad-hoc dicts.
  - `enums/` -- typed replacements for scipy's string-flag arguments (`ProjectionMethod`, `FiniteDifferenceMethod`, `HessianApproximationType`, `TrustConstrMethod`, ...).
- `src/test/java/de/labathome/optimization/` -- tests live in a different package (`de.labathome.optimization`) than the code under test. Each `Test<X>.java` targets one main-package class.

## Matrix library and the in-tree sparse module

All linear algebra is in-tree. The only production dependency is **`dev.ludovic.netlib`** (luhenry/netlib, 3.2.0) for BLAS/LAPACK kernels.

The matrix abstraction lives at **`de.labathome.trustconstr.matrix`** (8 files):

- `Matrix` -- minimal abstract base: shape (`getRowCount`/`getColumnCount`), element access (`getAsDouble`/`setAsDouble`), arithmetic (`mtimes`/`plus`/`minus`/`times`/`divide`/`transpose`/`abs`/`absInPlace`), norms (`norm2`/`normInf`), slicing (`subMatrix`/`selectColumns`), `solve(rhs)` (delegates to `LinAlg.solve`), and `rank()` (delegates to `LinAlg.svd`). `Matrix.Factory.*` is a thin compatibility surface that delegates to `DenseMatrix.*`. ~340 LoC.
- `DenseMatrix` -- concrete dense, backed by **column-major `double[]`** of length `rows x cols`. `data()` exposes the raw buffer for zero-copy LAPACK/BLAS calls. `mtimes(DenseMatrix)` dispatches straight to BLAS `dgemm`. Static factories: `zeros`, `eye`, `column`, `row`, `fromRows`, `fromColumnMajor`, `copyFromMatrix`. Arithmetic methods use covariant returns (`mtimes` returns `DenseMatrix`, etc.) so callers don't need to cast.
- `SparseMatrix` -- concrete sparse, backed by a Dictionary-of-Keys (`HashMap<Long, Double>`). Cheap `setAsDouble` build phase; `toCSR()` materialises to the CSR fast path for compute.
- `LinAlg` -- static facade over LAPACK and the BLAS rank updates: `solve(A,b)` (`dgesv`), `qr(A)` returning `QRResult` (`dgeqrf`+`dorgqr`), `svd(A)` returning `SVDResult` (`dgesvd`), `cholesky(A)` returning `CholResult` (`dpotrf`), `syr(A, alpha, x)` (`dsyr`, in-place rank-1, mirrors lower from upper), `syr2(A, alpha, x, y)` (`dsyr2`, in-place rank-2). Every routine takes `DenseMatrix.data()` directly -- no row-major<->column-major conversion.
- `QRResult`, `SVDResult`, `CholResult` -- Java records carrying the LAPACK output. `QRResult.solve(rhs)` does least-squares, `CholResult.solve(rhs)` reuses the captured Cholesky factor via `dpotrs`, `SVDResult.rank(tol)` and `reciprocalSingularValues()` cover the pseudo-inverse path.
- `MatrixOps` -- small static helpers on primitive arrays (`norm2(double[])`, `dot(double[], double[])`, `diag(double[])`).

The scipy-`sparse`-style fast path is at **`de.labathome.trustconstr.sparse`**:

- `CSRMatrix`, `CSCMatrix` -- primitive-array (`int[] indptr/indices`, `double[] data`) sparse storage matching scipy's `csr_array`/`csc_array`. `CSRMatrix.builder(rows, cols)` is the canonical way to build sparse incrementally -- DOK under the hood, compresses to CSR in `build()`.
- `SparseAssembly` -- `vstack`, `hstack`, `blockArray` (the `[[A,B],[C,D]]` shape from `projections.py:99`), and `assembleJacobianWithSlacks` (the optimised KKT-Jacobian build from `tr_interior_point.py:_assemble_sparse_jacobian`).
- `DenseSolve` -- LAPACK `dgetrf`/`dgetrs` wrapper for the KKT solve. Currently materialises the assembled CSR/CSC matrix to dense before factoring; replacing this with a true sparse LU is a single-call-site swap.
- `SparseLinearOperator` -- adapts a CSR/CSC matvec to the existing `interfaces.LinearOperator`.
- `CSRMatrix.fromMatrix(Matrix)` is the bridge into the CSR fast path; routes through `SparseMatrix.toCSR()` when the source is already sparse, otherwise materialises via `toDoubleArray()` then `fromDense`.

`LinearConstraint.jacEq`/`jacIneq` preserve sparsity end-to-end: when the user provides a sparse `A` (via `SparseMatrix.Factory.zeros(...)`), the row-selected output is built as a `SparseMatrix` rather than densifying through `Matrix.Factory.zeros`. `Projections.projections` detects `A.isSparse()` and routes through the AUGMENTED_SYSTEM factorization. `jacEqCSR()` / `jacIneqCSR()` build CSR directly from the sparse non-zeros (no dense intermediate). Asserted in `TestSparseLinearConstraint` (4 cases). `CombinedConstraint.jacEq`/`jacIneq` likewise auto-detect: if every contributing source produces a sparse part, the combined output is allocated as a `SparseMatrix`; if any source is dense, the combined output is dense. Asserted in `TestCombinedConstraint`.

## Porting conventions

- Methods that mutate a `State`/`StateIP` in scipy are translated as static methods on `MinimizeTrustConstr` that take the state as a parameter and return it (see `updateState`, `updateStateIP`).
- Keep Javadoc references to the scipy paper citations and equation numbers -- they're load-bearing for cross-referencing the algorithm.
- The development cadence (see `git log`) is "port one scipy function, write a parity test, commit." Match that -- don't bulk-port without tests.

## Public entry points

- `MinimizeTrustConstr.minimize(fun, grad, hess, x0, constraint, ...)` -- single-constraint entry. Accepts a `LinearConstraint`, `NonlinearConstraint`, or `null`. Routes equality-only / unconstrained problems to `EqualityConstrainedSQP` and any inequality problem to `TrustRegionInteriorPoint`. The result's `method` field reports which path was used.
- `MinimizeTrustConstr.minimize(fun, grad, x0, constraint, ...)` -- same as above but without analytic Hessian; uses a fresh BFGS strategy that the orchestrator updates from gradient deltas. Mirrors scipy's default behaviour (`hess` defaults to `BFGS()`).
- `MinimizeTrustConstr.minimize(fun, grad, strategy, x0, constraint, ...)` -- same as above but accepts any `HessianUpdateStrategy` (typically `BFGS.FACTORY.build()` or `SR1.FACTORY.build()`).
- `MinimizeTrustConstr.minimize(fun, x0, constraint, ...)` -- most hands-off entry: only the objective is supplied. Gradient comes from 2-point finite differences via `NumDiff.approxDerivative`; Hessian from BFGS.
- `MinimizeTrustConstr.minimize(fun, grad, hess, x0, Object[] constraints, ...)` -- multi-constraint entry. Combines via `CombinedConstraint` (concatenates eq rows then ineq rows across sources) before dispatching.
- `MinimizeTrustConstr.minimizeEqualityConstrained(...)` -- direct equality-constrained entry, used internally and by integration tests.
- `MinimizeTrustConstr.minimizeTrustConstr(...)` -- the full scipy-shaped API. **Body still incomplete past the prepared-constraint step**: canonical-form concatenation, Lagrangian Hessian assembly, and method dispatch are TODO.

## Constraint Hessian-of-Lagrangian

`NonlinearConstraint` accepts an optional `BiFunction<Matrix, Matrix, Matrix> hess(x, v)` -- the constraint Hessian-of-Lagrangian
{@code Sum_i v[i] * H_{c_i}(x)}. When supplied, the orchestrator combines it with the objective Hessian to form the full Lagrangian Hessian {@code H_obj + H_constraint}. `LinearConstraint` rows have zero Hessian (they're linear), so they contribute nothing here. `CombinedConstraint` walks its sources and slices `vEq`/`vIneq` into per-source segments before delegating.

For nonlinear problems where constraint curvature matters at the optimum (e.g. Maratos), supplying `hess` is essential -- without it the algorithm uses only the objective Hessian and convergence quality drops. With `hess` supplied, the Java port matches scipy iteration counts on Maratos (5 iterations vs scipy's 8 -- the difference is in initial-step heuristics, not algorithmic behaviour).

## Known gaps (work-in-progress)

- **`MinimizeTrustConstr.minimizeTrustConstr` body** is now an adapter over the raw equality / IP paths -- translates the scipy-shape signature (`BiFunction<Matrix, Object, ...>`, `Bounds`, `hessp`, `Object` constraints, etc.) into the typed forms accepted by `minimizeEqualityConstrainedRaw` / `minimizeInequalityConstrainedRaw`, folds `bounds` into the constraint set as a `LinearConstraint.fromBounds(...)`, and dispatches by which derivatives the caller supplied (analytic, BFGS-fallback, FD+BFGS-fallback). `hessp` is honored via `HessianLinearOperator` materialisation. **`callback`** is honored via `interfaces.IterationCallback` (returns `true` to terminate). The four **`initial*` tuning knobs** (`initialConstraintPenalty`, `initialTrustRadius`, `initialBarrierParameter`, `initialBarrierTolerance`) and **`factorizationMethod`** are plumbed through to `EqualityConstrainedSQP.eqSQP` and `TrustRegionInteriorPoint.trustRegionInteriorPoint`. **`finiteDifferenceRelStep`** flows through to `NumDiff` via the `buildFdGrad` helper. **`verbose`/`disp`** auto-install a printing callback that emits one ASCII-table line per iteration (`|niter|f evals|CG iter|f|tr radius|opt|c viol|`); a user-supplied callback takes precedence over the auto-printer; `disp=true` bumps `verbose` to 1 if it was 0 (mirrors scipy). **`barrierTol`** is now a separate parameter to the IP raw method -- the IP loop's xtol-based stopping branch is now `trustRadius < xtol && barrierParameter < barrierTol` (was `< gtol`); convenience overloads still pass `gtol` so existing behaviour is unchanged when users don't specify a separate value. **`sparseJacobian`** when explicitly set wraps every constraint in a `SparsityForcedConstraint` decorator that converts `jacEq`/`jacIneq` to the requested representation; when `Optional.empty()`, auto-detect logic in `LinearConstraint` and `CombinedConstraint` decides per call. When the caller leaves every adapter knob at its default, dispatch routes through the convenience overloads for the simplest path; when any is tweaked, dispatch routes directly to the raw paths with the user's values. Asserted in `TestMinimizeTrustConstrFullShape` (5 cases), `TestCallback` (4 cases), `TestTuningKnobs` (8 cases), and `TestSparseJacobianFlag` (3 cases: wrapper unit-checks, sparseJacobian=True forces sparse, sparseJacobian=False forces dense).
- **Scipy iteration-count parity** on the equality-constrained quadratic (3 iterations) and equality-constrained Rosenbrock (7 iterations) -- exact match. Both problems converge to machine precision on identical inputs to the scipy reference (verified via `src/test/python/regenerate_references.py`).
- **Multi-constraint, mixed eq+ineq, Bounds, nonlinear constraints with analytic Hessian** all working end-to-end. Test cases include the Maratos problem (Nocedal & Wright 15.4 -- converges in 5 iterations vs scipy's 8), `HyperbolicIneq` (N&W 15.1), `EqIneqRosenbrock`, `IneqRosenbrock`, `BoundedRosenbrock`, the Elec/Thomson problem with 2 electrons on the unit sphere (6 variables, 2 nonlinear inequality constraints, dense Hessian), `TestEmptyConstraint` (unit hyperbola), and Rosenbrock with active/inactive linear and nonlinear inequalities.
- **`NonlinearConstraint` finite-difference Jacobian fallback**: a `NonlinearConstraint(fun, lb, ub)` constructor (no analytic Jacobian) builds a 2-point FD Jacobian closure internally -- each `jacEq`/`jacIneq` call evaluates `fun` once for the baseline plus `n` more times for the perturbed columns. Mirrors scipy's behaviour when `jac` is omitted. Asserted in `TestNonlinearConstraintFiniteDiffJac` (unit hyperbola converges via FD; eval-count is `n+1`).
- **`keep_feasible=true` end-to-end**: every public `MinimizeTrustConstr.minimize(...)` entry calls `validateKeepFeasibleAtStart(constraint, x0)` before dispatch -- throws `IllegalArgumentException` if any row marked `keepFeasible[i]=true` is violated at the start. For inequality dispatch, the orchestrator now also extracts a per-canonical-ineq `boolean[] enforceFeasibility` from the constraint(s) (via `LinearConstraint.enforceFeasibilityIneq()`, `NonlinearConstraint.enforceFeasibilityIneq()`, `CombinedConstraint.enforceFeasibilityIneq()`) and passes it to `TrustRegionInteriorPoint`. `BarrierSubproblem.computeFunction` then drives the slack `s[i] = -cIneq[i]` for each enforced row, making the barrier function infinite when the iterate moves outside the kf row -- the trust-region step-acceptance rejects such steps. So mid-iteration kf enforcement is now wired up on the IP path (the equality path doesn't have an analogous mechanism since kf only applies to inequalities). Asserted in `TestKeepFeasibleValidation` (7 cases): the existing 5 start-validation cases plus `enforceFeasibilityIneqExtractsKfFromCanonicalRows` (two-sided interval row inherits kf on both ub and lb canonical rows) and `keepFeasibleFlowsToIpEnforceFeasibility` (active-bound Rosenbrock with kf=true converges).
- **Degenerate-Jacobian gracefully fails rather than crashing** (scipy `test_issue_18882`): a constraint whose Jacobian is effectively zero at the start (e.g. `c(u) = 1 + u_1^2/9 - u_2^2/16` with `u_0 = (0, 0)`) used to crash inside the SVD-based projection. Two fixes: (1) `Projections.projections` short-circuits when `A.normInf() < tolerance`, returning the same identity-nullspace + zero-LS / zero-rowspace operators as the truly-zero-row path; (2) the equality-path `lagrangianGrad` recovery wraps `(A A^T)^-^1` in a try/catch and falls back to `lagrangianGrad = grad` when the Jacobian is singular at termination. `OptimizeResult.success` is now derived from `(status == 1) || (status == 2 && constrViolation < gtol)`, so degenerate problems report `success = false` with `constraintViolation > 1e-8`. Asserted in `TestMinimizeConstrainedExtras.degenerateConstraintReportsFailure`.
- **Over-determined equality count rejected with helpful error** (scipy `test_gh20665`): when `nEq > nVars` (the QR / AugmentedSystem paths can't factor the resulting system), `MinimizeTrustConstr` throws `IllegalArgumentException` with a message naming the actual counts and suggesting `factorizationMethod=SVD_FACTORIZATION` as a workaround. The check is suppressed when SVD is selected (rank-revealing). Hooked into the typed-overload dispatcher and the `Object[]` multi-constraint dispatcher (which also validates the *combined* eq-row count across sources, not just per-source). Asserted in `TestMinimizeConstrainedExtras.overdeterminedEqualityRejectedWithHelpfulMessage`.
- **FD perturbations respect `Bounds(keep_feasible=True)`** (scipy `test_gh11649`): the full-shape adapter's FD-grad path now passes a `StrictBounds(bounds)` wrapper into `NumDiff.approxDerivative`, so finite-difference perturbations stay inside the supplied bounds. `LinearConstraint.fromBounds(bounds)` propagates `bounds.keepFeasible()` to the per-row `keepFeasible[]` on the resulting linear constraint, so the existing `validateKeepFeasibleAtStart` check fires when the start violates a bounded variable. `buildFdGrad` gained an optional `FiniteDifferenceBounds fdBounds` parameter -- `null` keeps the unbounded default for callers that don't have bounds. Asserted in `TestFdRespectsBounds` (2 cases: tracked-eval test confirms no FD evaluation leaves the bounds; infeasible-start test confirms the kf validator catches it). Note: this is the FD-bounds half of gh-11649; mid-iteration kf step-back-tracking inside the SQP/IP loop is still future work.
- **Unconstrained dispatch through trust-constr** (`MinimizeTrustConstr.minimize(..., null)`) routes through the equality path with an empty constraint set and converges. Two pieces wire it up: `Projections.projections` short-circuits on zero-row `A` (returns identity nullspace + empty LS / row-space operators), and `EqualityConstrainedSQP.safeNorm2` guards the SQP loop's normal-step / merit-function arithmetic from divisions by zero when the constraint Jacobian has shape 0xn. Asserted in `TestMinimizeConstrainedExtras.unconstrainedRosenbrock` (unconstrained Rosenbrock from `(0.5, 1.5)` converges in 17 iterations to machine precision).
- **Quasi-Newton (BFGS, SR1) and finite-difference fallbacks** working through the orchestrator on equality, active-bound, nonlinear-active, and unconstrained dispatch paths. `BFGS` and `FullHessianUpdateStrategy` carried column-vector dot-product bugs (`Mw.mtimes(w)` instead of `w.transpose().mtimes(Mw)`) that only surfaced once the orchestrator's no-Hessian overload exercised them; both fixed. `FiniteDifferenceOptions.FACTORY` had the same shared-singleton state-leak hazard as `ScalarFunction.FACTORY` -- the no-grad orchestrator overload now constructs a fresh `FiniteDifferenceOptionsFactory()` per call instead.
- **Scipy test-suite parity**: `test_projections.py`, `test_qp_subproblem.py`, and selected cases from `test_minimize_constrained.py` are not yet translated. `test_nested_minimize.py` (gh21193) is translated as `TestNestedMinimize`. The structural pieces of `test_canonical_constraint.py` are covered by `TestLinearConstraint`, `TestNonlinearConstraint` (including the multiplier-packing Hessian assertion), and `TestCombinedConstraint` (concatenation analog of scipy's `CanonicalConstraint.concatenate`). Note: our row order is row-major per source, not scipy's "uppers grouped then lowers grouped" 4-block ordering -- the algorithm is row-permutation-invariant so this divergence is internal-only and re-deriving the per-row reference values explicitly is enough to test parity. New scipy-reference values for Java integration tests live in `src/test/python/regenerate_references.py`.
- **`OptimizeResult.lagrangianGrad` and multiplier plumbing**: populated on both dispatch paths. The equality path uses least-squares multiplier recovery from the equality Jacobian. The interior-point path now exposes the augmented-system multiplier from the last barrier subproblem via `StatefulResult.v()` -- first `nEq` entries are equality multipliers, remaining `nIneq` entries equal the original problem's `lambda` (slack-row multipliers from the augmented system). The orchestrator slices and computes `g + Jeq^T v + Jineq^T lambda`. Active-bound and mixed eq+ineq tests assert `||lagrangianGrad||inf < 1e-3..1e-4` at the optimum (the sharp KKT-vanishing test). Eval counters are accurate on both paths via closures wrapping the user's fun/grad/hess at the orchestrator boundary -- `ScalarFunction`'s internal counters would otherwise read 0 because the closures bypass them.

## Bug fixes that landed alongside the orchestrator

A few latent bugs surfaced when wiring `EqualityConstrainedSQP` into a runnable orchestrator and were fixed:

- `EqualityConstrainedSQP.defaultScaling(n)` returned the input vector instead of the identity (the comment said "no scaling" but the body returned `x`). Now returns `Matrix.Factory.eye(n, n)`.
- `EqualityConstrainedSQP.eqSQP(...)` defaulted `trustUb` to `NEGATIVE_INFINITY` (typo). Now `POSITIVE_INFINITY`.
- `QPSubproblem.projectedCG`'s `reinforceBoxBoundaries` path went through `Matrix.Factory.importFromArray(double[])`, which produced a 1xn row matrix and silently broke shape consistency for downstream `dn + dt`. Now uses `UjmpBridge.arrayToCol(...)` which preserves the nx1 column convention. Four `TestProjectedCG` assertions had been written against the buggy row-shape; they were updated to column indexing.
- `ScalarFunction.ScalarFunctionFactory` constructor was private with a single shared `FACTORY` static instance whose state leaked between calls. Constructor is now `public` so callers can build a fresh factory per `minimize` invocation.
- `BarrierSubproblem.getScaling` placed the slack diagonal entries at offset `0` instead of `nVars`, producing a malformed scaling matrix that prevented `TrustRegionInteriorPoint` from converging. Now correctly populates the lower block.
- `Projections.projections` rejected `AUGMENTED_SYSTEM` for dense matrices (and `QR_FACTORIZATION` for sparse). Since the augmented-system path now goes through the in-tree CSR module regardless of input type, that gate has been removed.
- `EqualityConstrainedSQP.eqSQP` had a typo in the trust-region quadratic model: it computed `0.5 d.T H d + c.T c` (gradient norm squared) instead of `0.5 d.T H d + c.T d`. The bad term made the merit-function reduction ratio explode (~-1e14 at the first iteration on Maratos), driving the trust radius to collapse before the algorithm could make progress. Fixing this single character (`mtimes(c)` -> `mtimes(d)`) is what unblocked Maratos convergence and active-bound Rosenbrock convergence.
- `FullHessianUpdateStrategy.autoScale` and `BFGS.updateImplementation` both used `column.mtimes(column)` for what should have been a scalar dot product (`column.transpose().mtimes(column)`). The matrix multiply rejects the shape mismatch; the BFGS path errored out the moment a quasi-Newton update was attempted. The bug never surfaced because no caller exercised BFGS through the orchestrator until the no-Hessian `minimize` overload landed. Five sites fixed (3 in `autoScale`, 2 in `updateImplementation`).
