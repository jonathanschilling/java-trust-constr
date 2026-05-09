package org.scipy.optimize.minimize.interfaces;

import org.scipy.optimize.minimize.matrix.Matrix;

/**
 * Hessian of the Lagrangian {@code L(x, v) = f(x) + Sum v_i c_i(x)} with
 * respect to {@code x}, returned as a {@link LinearOperator} so callers can
 * apply it to vectors without materialising the full {@code n &times; n}
 * matrix.
 */
@FunctionalInterface
public interface LagrangeHessian {

	/**
	 * @param x current iterate ({@code n &times; 1})
	 * @param v full Lagrange-multiplier vector (equality + inequality)
	 * @return {@link LinearOperator} applying {@code grad^2L(x, v)} to a vector
	 */
	public LinearOperator lagrHess(Matrix x, Matrix v);
}
