package de.labathome.trustconstr.enums;

/**
 * How {@link de.labathome.trustconstr.Projections#projections} should
 * factor the constraint Jacobian to build the null-space / least-squares /
 * row-space projectors.
 */
public enum ProjectionMethod {

	/**
	 * The operators will be computed using the so-called normal equation approach
	 * explained in [1]_. In order to do so the Cholesky factorization of
	 * {@code (A A.T)} is computed. Exclusive for sparse matrices.
	 */
	NORMAL_EQUATION,

	/**
	 * The operators will be computed using the so-called augmented system approach
	 * explained in [1]_. Exclusive for sparse matrices.
	 */
	AUGMENTED_SYSTEM,

	/**
	 * Compute projections using QR factorization.
	 * Exclusive for dense matrices.
	 */
	QR_FACTORIZATION,

	/**
	 * Compute projections using SVD factorization.
	 * Exclusive for dense matrices.
	 */
	SVD_FACTORIZATION;
}
