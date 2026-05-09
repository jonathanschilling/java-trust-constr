# java-trust-constr

A Java port of `scipy.optimize.minimize(method='trust-constr')` -- a trust-region
constrained nonlinear optimizer that handles linear and nonlinear equality and
inequality constraints, and bounds.

Each Java class mirrors its scipy counterpart (e.g. `Projections.java` <->
`projections.py`); the upstream reference lives at
<https://github.com/scipy/scipy/tree/main/scipy/optimize/_trustregion_constr>.

## Getting it

Maven:

```xml
<dependency>
    <groupId>de.labathome</groupId>
    <artifactId>java-trust-constr</artifactId>
    <version>1.0.0</version>
</dependency>
```

Gradle:

```groovy
implementation 'de.labathome:java-trust-constr:1.0.0'
```

Requires Java 17 or newer. Pulls in `dev.ludovic.netlib:blas` and
`dev.ludovic.netlib:lapack` 3.2.0 transitively for BLAS / LAPACK kernels.

## Quick start

The public entry points live in
[`MinimizeTrustConstr`](src/main/java/de/labathome/trustconstr/MinimizeTrustConstr.java).
Pick the convenience overload that matches what you have:

### Analytic gradient and Hessian

```java
import de.labathome.trustconstr.MinimizeTrustConstr;
import de.labathome.trustconstr.LinearConstraint;
import de.labathome.trustconstr.records.OptimizeResult;
import de.labathome.trustconstr.matrix.DenseMatrix;
import de.labathome.trustconstr.matrix.Matrix;

// minimize x^2 + y^2  subject to  x + y == 2  ->  optimum at (1, 1)
java.util.function.Function<Matrix, Double> fun = x ->
        x.getAsDouble(0, 0) * x.getAsDouble(0, 0)
        + x.getAsDouble(1, 0) * x.getAsDouble(1, 0);

java.util.function.Function<Matrix, Matrix> grad = x ->
        DenseMatrix.column(
                2 * x.getAsDouble(0, 0),
                2 * x.getAsDouble(1, 0));

java.util.function.Function<Matrix, Matrix> hess = x ->
        DenseMatrix.fromRows(new double[][] {{2, 0}, {0, 2}});

LinearConstraint eq = new LinearConstraint(
        DenseMatrix.fromRows(new double[][] {{1.0, 1.0}}),
        new double[] {2.0}, new double[] {2.0});

Matrix x0 = DenseMatrix.column(-1.0, 0.5);
OptimizeResult r = MinimizeTrustConstr.minimize(
        fun, grad, hess, x0, eq, /*maxIter=*/ 200, /*xtol=*/ 1e-10, /*gtol=*/ 1e-10);
System.out.println(r);
// OptimizeResult{success=true, status=1, method=EQUALITY_CONSTRAINED_SQP,
//   x=[1.00000, 1.00000], fun=2.00000, optimality=...}
```

### Quasi-Newton fallback (BFGS)

Omit the Hessian and pass only `fun` and `grad`. The orchestrator builds a
fresh BFGS update strategy per call, mirroring scipy's default when `hess` is
omitted.

```java
OptimizeResult r = MinimizeTrustConstr.minimize(fun, grad, x0, eq, 500, 1e-8, 1e-8);
```

To pick a specific strategy:

```java
import de.labathome.trustconstr.SR1;

OptimizeResult r = MinimizeTrustConstr.minimize(
        fun, grad, SR1.FACTORY.build(), x0, eq, 500, 1e-8, 1e-8);
```

### Most hands-off: only the objective

Both gradient and Hessian are omitted -- the orchestrator computes the gradient
by 2-point finite differences and the Hessian by BFGS:

```java
OptimizeResult r = MinimizeTrustConstr.minimize(fun, x0, eq, 500, 1e-6, 1e-6);
```

### Inequality / nonlinear / multi-constraint

Pass a `LinearConstraint`, a `NonlinearConstraint`, or `null` to the typed
overload. For multiple constraints, pass an `Object[]`:

```java
import de.labathome.trustconstr.NonlinearConstraint;

// x^2 - y^2 >= 1 (unit hyperbola), with constraint Hessian-of-Lagrangian.
java.util.function.Function<Matrix, Matrix> cFun = x ->
        DenseMatrix.column(
                x.getAsDouble(0, 0) * x.getAsDouble(0, 0)
                - x.getAsDouble(1, 0) * x.getAsDouble(1, 0));
java.util.function.Function<Matrix, Matrix> cJac = x ->
        DenseMatrix.fromRows(new double[][] {
                {2 * x.getAsDouble(0, 0), -2 * x.getAsDouble(1, 0)}});
java.util.function.BiFunction<Matrix, Matrix, Matrix> cHess = (x, v) -> {
    double v0 = v.getAsDouble(0, 0);
    return DenseMatrix.fromRows(new double[][] {{2 * v0, 0}, {0, -2 * v0}});
};
NonlinearConstraint c = new NonlinearConstraint(cFun, cJac, cHess,
        new double[] {1.0}, new double[] {Double.POSITIVE_INFINITY}, null);

OptimizeResult r = MinimizeTrustConstr.minimize(fun, grad, hess,
        DenseMatrix.column(1.5, 0.0), c,
        1000, 1e-8, 1e-8);
```

If `jac` is omitted, `NonlinearConstraint` builds a 2-point finite-difference
Jacobian internally.

### Full scipy-shape API

For users who want every knob -- `args`, `Bounds`, `hessp`, callback,
`finiteDifferenceRelStep`, `factorizationMethod`, the four `initial*` tuning
parameters, and `verbose`/`disp` console output -- use
`MinimizeTrustConstr.minimizeTrustConstr(...)`. See its Javadoc for the full
signature.

```java
import de.labathome.trustconstr.interfaces.IterationCallback;

IterationCallback log = state -> {
    System.out.printf("iter=%d f=%g opt=%g%n",
            state.nIter, state.fun, state.optimality);
    return state.optimality < 1e-8;   // request termination
};

OptimizeResult r = MinimizeTrustConstr.minimizeTrustConstr(
        (x, args) -> fun.apply(x), x0, /*args=*/ null,
        (x, args) -> grad.apply(x), (x, args) -> hess.apply(x),
        /*hessp=*/ null, /*bounds=*/ null, eq,
        /*xtol=*/ 1e-10, /*gtol=*/ 1e-10, /*barrierTol=*/ 1e-10,
        /*sparseJacobian=*/ java.util.Optional.empty(),
        log, /*maxIter=*/ 1000, /*verbose=*/ 0, /*finiteDifferenceRelStep=*/ null,
        /*initialConstraintPenalty=*/ 1.0, /*initialTrustRadius=*/ 1.0,
        /*initialBarrierParameter=*/ 0.1, /*initialBarrierTolerance=*/ 0.1,
        /*factorizationMethod=*/ null, /*disp=*/ false);
```

## Result

[`OptimizeResult`](src/main/java/de/labathome/trustconstr/records/OptimizeResult.java)
exposes the scipy-style fields:

| Field                       | Meaning                                                |
|-----------------------------|--------------------------------------------------------|
| `success`                   | `true` if the run converged                            |
| `x`                         | Solution vector                                        |
| `fun`                       | Objective at the solution                              |
| `grad`, `lagrangianGrad`    | Gradient and Lagrangian gradient at the solution       |
| `optimality`                | Infinity norm of the Lagrangian gradient               |
| `constraintViolation`       | Maximum residual constraint violation                  |
| `nIter`, `cgIter`           | Outer / inner iteration counts                         |
| `numFunctionEval`, etc.     | Evaluation counters for `fun` / `grad` / `hess`        |
| `status`, `message`         | Termination status code (0..3) and human-readable reason |
| `method`                    | Which dispatch path ran (`EQUALITY_CONSTRAINED_SQP` or `TRUST_REGION_INTERIOR_POINT`) |
| `trustRadius`               | Final trust-region radius                              |
| `barrierParameter`, `barrierTolerance` | IP-path only -- final values from the last barrier subproblem |

`println(result)` prints a one-line summary.

## Javadoc

API reference: <https://jonathanschilling.github.io/java-trust-constr/>.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release notes.

---

## Architecture

See [CLAUDE.md](CLAUDE.md) for codebase architecture: the `Projections /
QPSubproblem / EqualityConstrainedSQP / BarrierSubproblem /
TrustRegionInteriorPoint` decomposition, the in-tree `sparse` module
(`CSRMatrix`, `CSCMatrix`, `SparseAssembly`, `DenseSolve`), known gaps, and the
catalogue of integration tests.

---

## Building from source

Maven, Java 17.

```bash
mvn compile
mvn test
```

The parent POM `de.labathome:de-labathome-parent:1.1.0` is on Maven Central,
so a fresh `mvn` resolves everything from a public mirror -- no local install
of the parent is needed.

The Python helper at `src/test/python/regenerate_references.py` produces
the reference values used by integration-test fixtures; it depends on a
pip-installed `scipy` (`pip install scipy`), not on the upstream sources.

## Continuous integration

Two GitHub Actions workflows:

- **`.github/workflows/ci.yml`** -- runs `mvn -B test` plus a javadoc lint
  check and an ASCII-only invariant check on every push to `master` and on
  every pull request.
- **`.github/workflows/javadoc.yml`** -- rebuilds the javadoc and pushes it
  to the `gh-pages` branch on every push to `master` and every `v*` release
  tag. The published artefact is at
  <https://jonathanschilling.github.io/java-trust-constr/>.

## Publishing to Maven Central

The legacy OSSRH service (`oss.sonatype.org`) was shut down on 2025-06-30; the
parent POM was modernised to use the **Central Portal**
(`central.sonatype.com`) via the official
`org.sonatype.central:central-publishing-maven-plugin`.

One-time setup:

1. **Create / claim the namespace** at <https://central.sonatype.com>. The
   `de.labathome` namespace must show up under "View Namespaces" -- if it
   doesn't (e.g. you signed up after the OSSRH sunset and never published
   under this namespace), submit it and add the issued `OSSRH-NNNNN`
   verification key as a `TXT` record on the apex `labathome.de` domain.
2. **Generate a user token** (Portal -> "Generate User Token"). This produces a
   `username`/`password` pair shown once; if lost, regenerate.
3. **Add a `<server>` entry** to `~/.m2/settings.xml`:

   ```xml
   <server>
     <id>central</id>
     <username><!-- token username --></username>
     <password><!-- token password --></password>
   </server>
   ```

4. **GPG**: have a primary signing key with no `usage: S` sub-key (Central
   only verifies against the primary key; sub-keys silently fail). Push the
   public key to `keyserver.ubuntu.com`:

   ```bash
   gpg --keyserver keyserver.ubuntu.com --send-keys <KEYID>
   ```

   Add `<server id="...">` plus `<gpg.keyname>` to `settings.xml` so the
   passphrase isn't prompted interactively.

Then publish a release:

```bash
mvn clean deploy -P release
```

The `release` profile in the parent POM activates `maven-gpg-plugin` 3.2.7
and `central-publishing-maven-plugin` 0.10.0, which uploads the validated
bundle to the Portal. With `<autoPublish>false</autoPublish>` (the default
in the parent), the bundle waits for a manual "Publish" click in the Portal
UI; flip to `true` to release as soon as validation passes.
