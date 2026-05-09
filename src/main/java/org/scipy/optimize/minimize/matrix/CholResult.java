package org.scipy.optimize.minimize.matrix;

import dev.ludovic.netlib.lapack.LAPACK;
import org.netlib.util.intW;

/**
 * Cholesky factorisation of a symmetric positive-definite matrix.
 * {@code factor} holds the LAPACK {@code dpotrf} output verbatim -- the upper
 * (if {@code upper}) or lower triangle stores the Cholesky factor; the other
 * triangle is unspecified. Reuse the captured factor for repeated solves
 * via {@link #solve(DenseMatrix)} (one {@code dpotrs} call, no re-factoring).
 */
public record CholResult(double[] factor, int n, boolean upper) {

	/**
	 * Solve {@code A x = rhs} where {@code A} is the original SPD matrix
	 * factored into {@link #factor()}. Multiple right-hand sides supported
	 * via the column count of {@code rhs}. Does not modify {@link #factor()}.
	 *
	 * @param rhs right-hand side ({@code n x nrhs}); must have the same row
	 *            count as the factored matrix
	 * @return the solution {@code x} ({@code n x nrhs})
	 */
	public DenseMatrix solve(DenseMatrix rhs) {
		if (rhs.rows() != n) {
			throw new IllegalArgumentException("CholResult.solve: rhs row count "
					+ rhs.rows() + " != factor dimension " + n);
		}
		int nrhs = rhs.cols();
		double[] bFlat = rhs.data().clone();
		intW info = new intW(0);
		LAPACK.getInstance().dpotrs(upper ? "U" : "L", n, nrhs, factor, n, bFlat, n, info);
		if (info.val != 0) {
			throw new ArithmeticException("dpotrs failed: info=" + info.val);
		}
		return DenseMatrix.fromColumnMajor(n, nrhs, bFlat);
	}
}
