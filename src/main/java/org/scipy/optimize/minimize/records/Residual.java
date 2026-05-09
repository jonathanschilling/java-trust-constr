package org.scipy.optimize.minimize.records;

/**
 * Lower and upper residuals for a two-sided constraint
 * {@code lb &le; f(x) &le; ub}. Returned by
 * {@link org.scipy.optimize.minimize.LinearConstraint#residual(org.scipy.optimize.minimize.matrix.Matrix)}
 * and {@link Bounds#residual(org.scipy.optimize.minimize.matrix.Matrix)}.
 *
 * <p>Mirrors scipy's tuple return shape from
 * {@code LinearConstraint.residual(x)} and {@code Bounds.residual(x)}:
 *
 * <pre>
 *     sl = f(x) - lb     (non-negative iff lower bound is satisfied)
 *     sb = ub - f(x)     (non-negative iff upper bound is satisfied)
 * </pre>
 *
 * <p>Both components are non-negative iff every row of the constraint is
 * satisfied. A negative entry indicates the corresponding row is violated.
 *
 * @param sl lower residual ({@code f(x) - lb})
 * @param sb upper residual ({@code ub - f(x)})
 */
public record Residual(double[] sl, double[] sb) {}
