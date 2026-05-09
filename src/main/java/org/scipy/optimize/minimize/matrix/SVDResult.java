package org.scipy.optimize.minimize.matrix;

/**
 * Singular-value decomposition {@code A = U · diag(s) · Vᵀ}.
 *
 * <p>Shapes:
 * <ul>
 *   <li>{@code U} is m × m</li>
 *   <li>{@code s} has length {@code min(m, n)}; values are non-negative and
 *       monotonically non-increasing (LAPACK convention)</li>
 *   <li>{@code Vt} is n × n (already transposed — i.e. {@code Vᵀ} from the
 *       textbook decomposition; rows are right-singular vectors)</li>
 * </ul>
 */
public record SVDResult(DenseMatrix U, double[] s, DenseMatrix Vt) {

	/**
	 * Number of singular values strictly greater than {@code tol}. Use the
	 * LAPACK convention {@code tol = max(m,n) · ulp(σ_max)} for numerical
	 * rank if no problem-specific cutoff is known.
	 */
	public int rank(double tol) {
		int r = 0;
		for (double sv : s) if (sv > tol) ++r;
		return r;
	}

	/**
	 * Element-wise reciprocal of the singular values, with values at or below
	 * the LAPACK numerical-zero threshold {@code max(m,n) · ulp(σ_max)} mapped
	 * to {@code 0}. Use this when forming {@code A⁺ = V · diag(1/s) · Uᵀ}.
	 */
	public double[] reciprocalSingularValues() {
		int m = (int) U.rows();
		int n = (int) Vt.cols();
		double sigmaMax = s.length > 0 ? s[0] : 0.0;
		double tol = Math.max(m, n) * Math.ulp(sigmaMax);
		double[] out = new double[s.length];
		for (int i = 0; i < s.length; ++i) {
			out[i] = s[i] > tol ? 1.0 / s[i] : 0.0;
		}
		return out;
	}
}
