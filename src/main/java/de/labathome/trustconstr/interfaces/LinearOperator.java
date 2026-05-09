package de.labathome.trustconstr.interfaces;

import de.labathome.trustconstr.matrix.Matrix;

/**
 * Matrix-free linear operator: applies an underlying linear map to a vector
 * without materialising the matrix. Mirrors scipy's
 * {@code scipy.sparse.linalg.LinearOperator}.
 */
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
