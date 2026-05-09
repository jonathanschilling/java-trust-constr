package org.scipy.optimize.minimize.interfaces;

import org.scipy.optimize.minimize.matrix.Matrix;

@FunctionalInterface
public interface LinearOperator {

	/**
	 * Apply the linear operator to a given vector.
	 *
	 * @param x [n] vector
	 * @return [m] result
	 */
	public Matrix apply(Matrix x);
}
