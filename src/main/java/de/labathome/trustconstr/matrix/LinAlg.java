package de.labathome.trustconstr.matrix;

import dev.ludovic.netlib.blas.BLAS;
import dev.ludovic.netlib.lapack.LAPACK;
import org.netlib.util.intW;

/**
 * Thin static facade over {@code dev.ludovic.netlib} BLAS / LAPACK kernels,
 * specialised to {@link DenseMatrix} (column-major {@code double[]}). All
 * routines pass {@link DenseMatrix#data()} straight to the native call -- no
 * row-major / column-major conversion, no extra allocation beyond the working
 * buffer that LAPACK overwrites.
 *
 * <p>Each routine documents whether it modifies the input. Where it does, the
 * implementation copies the buffer first so callers don't need to defensively
 * copy at every site.
 */
public final class LinAlg {

	private static final BLAS BLAS_INSTANCE = BLAS.getInstance();
	private static final LAPACK LAPACK_INSTANCE = LAPACK.getInstance();

	private LinAlg() {}

	// ----- LU solve -----

	/**
	 * Solve {@code A x = b} via LAPACK {@code dgesv} (general LU with partial
	 * pivoting). {@code A} must be square. Inputs are not modified -- the
	 * implementation operates on copies.
	 *
	 * @param A square coefficient matrix
	 * @param b right-hand side ({@code A.rows x nrhs}); supports multiple RHS
	 * @return the solution {@code x} ({@code A.rows x nrhs})
	 */
	public static DenseMatrix solve(DenseMatrix A, DenseMatrix b) {
		int n = A.rows();
		if (A.cols() != n) {
			throw new IllegalArgumentException("solve: A must be square; got "
					+ n + "x" + A.cols());
		}
		if (b.rows() != n) {
			throw new IllegalArgumentException("solve: b row count " + b.rows()
					+ " != A dimension " + n);
		}
		int nrhs = b.cols();
		double[] aFlat = A.data().clone();
		double[] bFlat = b.data().clone();
		int[] ipiv = new int[n];
		intW info = new intW(0);
		LAPACK_INSTANCE.dgesv(n, nrhs, aFlat, n, ipiv, bFlat, n, info);
		if (info.val != 0) {
			throw new ArithmeticException("dgesv failed: info=" + info.val);
		}
		return DenseMatrix.fromColumnMajor(n, nrhs, bFlat);
	}

	// ----- QR factorisation -----

	/**
	 * QR factorisation via {@code dgeqrf} + {@code dorgqr}. {@code A} must
	 * be tall-or-square (m >= n).
	 *
	 * @param A {@code m x n} matrix to factor
	 * @return {@link QRResult} with {@code Q} ({@code m x n}, orthonormal
	 *         columns) and {@code R} ({@code n x n}, upper triangular)
	 */
	public static QRResult qr(DenseMatrix A) {
		int m = A.rows();
		int n = A.cols();
		if (m < n) {
			throw new IllegalArgumentException("qr requires m >= n; got " + m + "x" + n);
		}
		double[] flat = A.data().clone();
		double[] tau = new double[Math.min(m, n)];

		// dgeqrf workspace query then factor.
		intW info = new intW(0);
		double[] work = new double[1];
		LAPACK_INSTANCE.dgeqrf(m, n, flat, m, tau, work, -1, info);
		int lwork = Math.max(1, (int) work[0]);
		work = new double[lwork];
		LAPACK_INSTANCE.dgeqrf(m, n, flat, m, tau, work, lwork, info);
		if (info.val != 0) {
			throw new ArithmeticException("dgeqrf failed: info=" + info.val);
		}

		// Extract R from the upper triangle of the packed factorisation.
		DenseMatrix R = new DenseMatrix(n, n);
		for (int j = 0; j < n; ++j) {
			for (int i = 0; i <= j; ++i) {
				R.set(i, j, flat[i + j * m]);
			}
		}

		// dorgqr in-place: reconstruct Q, overwriting flat.
		intW info2 = new intW(0);
		double[] work2 = new double[1];
		LAPACK_INSTANCE.dorgqr(m, n, n, flat, m, tau, work2, -1, info2);
		int lwork2 = Math.max(1, (int) work2[0]);
		work2 = new double[lwork2];
		LAPACK_INSTANCE.dorgqr(m, n, n, flat, m, tau, work2, lwork2, info2);
		if (info2.val != 0) {
			throw new ArithmeticException("dorgqr failed: info=" + info2.val);
		}

		// Q is m x n column-major in `flat[0..m*n-1]`.
		double[] qData = new double[m * n];
		System.arraycopy(flat, 0, qData, 0, m * n);
		DenseMatrix Q = DenseMatrix.fromColumnMajor(m, n, qData);

		return new QRResult(Q, R);
	}

	// ----- SVD -----

	/**
	 * Full SVD via {@code dgesvd} ({@code jobu='A'}, {@code jobvt='A'}):
	 * {@code A = U * diag(s) * V^T}.
	 *
	 * @param A {@code m x n} matrix to factor
	 * @return {@link SVDResult} with {@code U} ({@code m x m}),
	 *         {@code s} (length {@code min(m, n)}, monotonically
	 *         non-increasing), and {@code V^T} ({@code n x n})
	 */
	public static SVDResult svd(DenseMatrix A) {
		int m = A.rows();
		int n = A.cols();
		double[] aFlat = A.data().clone();
		double[] s = new double[Math.min(m, n)];
		double[] uFlat = new double[m * m];
		double[] vtFlat = new double[n * n];

		intW info = new intW(0);
		double[] work = new double[1];
		LAPACK_INSTANCE.dgesvd("A", "A", m, n, aFlat, m, s, uFlat, m, vtFlat, n, work, -1, info);
		int lwork = Math.max(1, (int) work[0]);
		work = new double[lwork];
		LAPACK_INSTANCE.dgesvd("A", "A", m, n, aFlat, m, s, uFlat, m, vtFlat, n, work, lwork, info);
		if (info.val != 0) {
			throw new ArithmeticException("dgesvd failed: info=" + info.val);
		}
		return new SVDResult(
				DenseMatrix.fromColumnMajor(m, m, uFlat),
				s,
				DenseMatrix.fromColumnMajor(n, n, vtFlat));
	}

	// ----- Cholesky -----

	/**
	 * Cholesky factorisation of an SPD matrix via {@code dpotrf}, upper
	 * triangle. Input is not modified -- implementation operates on a copy.
	 *
	 * @param A square SPD matrix
	 * @return {@link CholResult} carrying the upper-triangular Cholesky
	 *         factor in LAPACK column-major layout
	 */
	public static CholResult cholesky(DenseMatrix A) {
		int n = A.rows();
		if (A.cols() != n) {
			throw new IllegalArgumentException("cholesky: A must be square; got "
					+ n + "x" + A.cols());
		}
		double[] flat = A.data().clone();
		intW info = new intW(0);
		LAPACK_INSTANCE.dpotrf("U", n, flat, n, info);
		if (info.val != 0) {
			throw new ArithmeticException("dpotrf failed: info=" + info.val
					+ " (matrix is not SPD)");
		}
		return new CholResult(flat, n, true);
	}

	// ----- Symmetric rank updates (in-place, on the upper triangle) -----

	/**
	 * In-place rank-1 symmetric update: {@code A := A + alpha x x^T}. {@code A}
	 * must be square; only the upper triangle is updated by BLAS, after which
	 * the lower triangle is mirrored from the upper to keep {@code A}
	 * symmetric.
	 *
	 * @param A     square matrix to update in place ({@code n x n})
	 * @param alpha scalar multiplier
	 * @param x     length-{@code n} vector
	 */
	public static void syr(DenseMatrix A, double alpha, double[] x) {
		int n = A.rows();
		if (A.cols() != n) {
			throw new IllegalArgumentException("syr: A must be square; got "
					+ n + "x" + A.cols());
		}
		if (x.length != n) {
			throw new IllegalArgumentException("syr: x length " + x.length + " != n " + n);
		}
		BLAS_INSTANCE.dsyr("U", n, alpha, x, 1, A.data(), n);
		mirrorUpperToLower(A);
	}

	/**
	 * In-place rank-2 symmetric update: {@code A := A + alpha (x y^T + y x^T)}.
	 * {@code A} must be square; only the upper triangle is updated, and the
	 * lower triangle is mirrored from the upper afterwards.
	 *
	 * @param A     square matrix to update in place ({@code n x n})
	 * @param alpha scalar multiplier
	 * @param x     length-{@code n} vector
	 * @param y     length-{@code n} vector
	 */
	public static void syr2(DenseMatrix A, double alpha, double[] x, double[] y) {
		int n = A.rows();
		if (A.cols() != n) {
			throw new IllegalArgumentException("syr2: A must be square; got "
					+ n + "x" + A.cols());
		}
		if (x.length != n || y.length != n) {
			throw new IllegalArgumentException("syr2: x/y length must equal n=" + n
					+ "; got x=" + x.length + " y=" + y.length);
		}
		BLAS_INSTANCE.dsyr2("U", n, alpha, x, 1, y, 1, A.data(), n);
		mirrorUpperToLower(A);
	}

	private static void mirrorUpperToLower(DenseMatrix A) {
		int n = A.rows();
		double[] d = A.data();
		// Column-major: A[i + j*n] is row i, col j.
		// "U" upper triangle in Fortran is i <= j (i.e. row index <= col index).
		// Mirror upper -> lower: for i < j, set d[j + i*n] = d[i + j*n].
		for (int j = 1; j < n; ++j) {
			for (int i = 0; i < j; ++i) {
				d[j + i * n] = d[i + j * n];
			}
		}
	}
}
