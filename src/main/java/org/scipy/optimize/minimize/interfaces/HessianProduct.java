package org.scipy.optimize.minimize.interfaces;

import org.scipy.optimize.minimize.matrix.Matrix;

/**
 * A matrix-free Hessian: given the current point {@code x} and a vector
 * {@code p}, return {@code H(x) p}.
 *
 * Mirrors the {@code hessp(x, p, *args)} callable from scipy's
 * {@code minimize(method='trust-constr')} API. Pairs with
 * {@link org.scipy.optimize.minimize.HessianLinearOperator}, which adapts a
 * {@code HessianProduct} to the explicit-matrix slot used elsewhere in the
 * port.
 */
@FunctionalInterface
public interface HessianProduct {

	/**
	 * @param x    [n] current point
	 * @param p    [n] vector to multiply
	 * @param args optional extra arguments forwarded by the caller
	 * @return     [n] result of {@code H(x) p}
	 */
	Matrix apply(Matrix x, Matrix p, Object args);
}
