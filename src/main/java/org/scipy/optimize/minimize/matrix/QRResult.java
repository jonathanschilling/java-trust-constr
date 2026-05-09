package org.scipy.optimize.minimize.matrix;

/**
 * QR factorisation of a tall-or-square matrix {@code A} (m × n with m ≥ n):
 * {@code A = Q R} with {@code Q} (m × n) having orthonormal columns and
 * {@code R} (n × n) upper triangular.
 */
public record QRResult(DenseMatrix Q, DenseMatrix R) {

	/**
	 * Least-squares solve: returns the {@code n × p} matrix {@code x} that
	 * minimises {@code ‖A x − rhs‖₂}, where {@code A = Q R} was the original
	 * input. Computes {@code x = R⁻¹ Qᵀ rhs}.
	 */
	public DenseMatrix solve(DenseMatrix rhs) {
		DenseMatrix qtRhs = (DenseMatrix) Q.transpose().mtimes(rhs);
		return LinAlg.solve(R, qtRhs);
	}
}
