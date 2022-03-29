package org.scipy.optimize.minimize.interfaces;

import org.ujmp.core.Matrix;

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
