package org.scipy.optimize.minimize.records;

import org.scipy.optimize.minimize.matrix.Matrix;

/**
 * Pair of canonical equality and inequality components produced by a
 * {@link org.scipy.optimize.minimize.CanonicalConstraint}'s {@code fun(x)}
 * or {@code jac(x)}. Mirrors scipy's tuple return shape from
 * {@code canonical_constraint.py}.
 *
 * <p>For {@code fun(x)}: {@code eq} is {@code nEq x 1}, {@code ineq} is
 * {@code nIneq x 1} (column vectors of canonical residuals).
 *
 * <p>For {@code jac(x)}: {@code eq} is {@code nEq x n}, {@code ineq} is
 * {@code nIneq x n} (canonical Jacobian blocks).
 *
 * @param eq   equality component
 * @param ineq inequality component
 */
public record EqIneqSplit(Matrix eq, Matrix ineq) {}
