package org.scipy.optimize.minimize.matrix;

import dev.ludovic.netlib.lapack.LAPACK;
import org.netlib.util.intW;

/**
 * QR decomposition of a tall-or-square matrix {@code A} (m × n with
 * m ≥ n). Computes {@code A = Q R} with {@code Q} (m × n) having
 * orthonormal columns and {@code R} (n × n) upper triangular.
 *
 * <p>API matches the slice of UJMP's {@code QRMatrix} that
 * {@link org.scipy.optimize.minimize.Projections} relies on:
 * {@link #getQ()} and {@link #getR()}. Backed by LAPACK
 * {@code dgeqrf} + {@code dorgqr}.
 */
public final class QRMatrix {

	private final DMatrix Q;
	private final DMatrix R;

	public QRMatrix(Matrix a) {
		int m = (int) a.getRowCount();
		int n = (int) a.getColumnCount();
		if (m < n) {
			throw new IllegalArgumentException("QRMatrix requires m >= n; got " + m + "x" + n);
		}
		double[] flat = MatrixIO.toColumnMajor(a);
		double[] tau = new double[Math.min(m, n)];
		intW info = new intW(0);
		double[] work = new double[1];
		LAPACK lapack = LAPACK.getInstance();
		// Workspace query then factorise.
		lapack.dgeqrf(m, n, flat, m, tau, work, -1, info);
		int lwork = Math.max(1, (int) work[0]);
		work = new double[lwork];
		lapack.dgeqrf(m, n, flat, m, tau, work, lwork, info);
		if (info.val != 0) {
			throw new ArithmeticException("dgeqrf failed: info=" + info.val);
		}

		// Extract R from upper triangle of flat.
		DMatrix Rmat = new DMatrix(n, n);
		for (int j = 0; j < n; ++j) {
			for (int i = 0; i <= j; ++i) {
				Rmat.set(i, j, flat[i + j * m]);
			}
		}

		// Materialise Q (m × n) by overwriting flat via dorgqr.
		intW info2 = new intW(0);
		double[] work2 = new double[1];
		lapack.dorgqr(m, n, n, flat, m, tau, work2, -1, info2);
		int lwork2 = Math.max(1, (int) work2[0]);
		work2 = new double[lwork2];
		lapack.dorgqr(m, n, n, flat, m, tau, work2, lwork2, info2);
		if (info2.val != 0) {
			throw new ArithmeticException("dorgqr failed: info=" + info2.val);
		}
		DMatrix Qmat = new DMatrix(m, n);
		for (int j = 0; j < n; ++j) {
			for (int i = 0; i < m; ++i) {
				Qmat.set(i, j, flat[i + j * m]);
			}
		}

		this.Q = Qmat;
		this.R = Rmat;
	}

	public Matrix getQ() { return Q; }
	public Matrix getR() { return R; }

	/**
	 * Least-squares solve: returns the {@code n × p} matrix {@code x} that
	 * minimises {@code ||A x - rhs||₂} (where {@code A} was the input to
	 * {@link #QRMatrix(Matrix)}). Computes {@code x = R⁻¹ Qᵀ rhs}.
	 */
	public Matrix solve(Matrix rhs) {
		Matrix qtRhs = Q.transpose().mtimes(rhs);
		// R is upper triangular n × n. Backsubstitute via the generic LU solve
		// (correct, just slower than dtrtrs).
		return R.solve(qtRhs);
	}
}
