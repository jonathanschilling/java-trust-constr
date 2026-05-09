package de.labathome.trustconstr.matrix;

/**
 * QR factorisation of a tall-or-square matrix {@code A} (m x n with m >= n):
 * {@code A = Q R} with {@code Q} (m x n) having orthonormal columns and
 * {@code R} (n x n) upper triangular.
 */
public record QRResult(DenseMatrix Q, DenseMatrix R) {

	/**
	 * Least-squares solve: returns the {@code n x p} matrix {@code x} that
	 * minimises {@code ||A x - rhs||_2}, where {@code A = Q R} was the original
	 * input. Computes {@code x = R^-^1 Q^T rhs}.
	 *
	 * @param rhs right-hand side ({@code m x p})
	 * @return least-squares solution ({@code n x p})
	 */
	public DenseMatrix solve(DenseMatrix rhs) {
		DenseMatrix qtRhs = (DenseMatrix) Q.transpose().mtimes(rhs);
		return LinAlg.solve(R, qtRhs);
	}
}
