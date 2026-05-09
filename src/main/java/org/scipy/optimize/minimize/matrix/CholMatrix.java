package org.scipy.optimize.minimize.matrix;

import dev.ludovic.netlib.lapack.LAPACK;
import org.netlib.util.intW;

/**
 * Cholesky decomposition of a symmetric positive-definite matrix {@code A}
 * via LAPACK {@code dpotrf}. Provides {@link #solve(Matrix)} for repeated
 * RHS solves through {@code dpotrs}, mirroring the slice of UJMP's
 * {@code CholMatrix} used by
 * {@link org.scipy.optimize.minimize.Projections#normalEquationProjections}.
 */
public final class CholMatrix {

	private final int n;
	private final double[] flat;

	public CholMatrix(Matrix a) {
		int rows = (int) a.getRowCount();
		int cols = (int) a.getColumnCount();
		if (rows != cols) {
			throw new IllegalArgumentException("CholMatrix requires square input; got " + rows + "x" + cols);
		}
		this.n = rows;
		this.flat = MatrixIO.toColumnMajor(a);
		intW info = new intW(0);
		LAPACK.getInstance().dpotrf("U", n, flat, n, info);
		if (info.val != 0) {
			throw new ArithmeticException("dpotrf failed: info=" + info.val
					+ " (matrix not positive definite)");
		}
	}

	public Matrix solve(Matrix rhs) {
		int nrhs = (int) rhs.getColumnCount();
		if (rhs.getRowCount() != n) {
			throw new IllegalArgumentException("solve(): RHS row count must match factor size");
		}
		double[] bFlat = MatrixIO.toColumnMajor(rhs);
		intW info = new intW(0);
		LAPACK.getInstance().dpotrs("U", n, nrhs, flat, n, bFlat, n, info);
		if (info.val != 0) {
			throw new ArithmeticException("dpotrs failed: info=" + info.val);
		}
		return MatrixIO.fromColumnMajor(bFlat, n, nrhs);
	}
}
