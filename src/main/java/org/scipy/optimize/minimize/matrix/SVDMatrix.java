package org.scipy.optimize.minimize.matrix;

import dev.ludovic.netlib.lapack.LAPACK;
import org.netlib.util.intW;

/**
 * Singular value decomposition {@code A = U S V^T}. Provides the slice of
 * UJMP's {@code SVDMatrix} that {@link org.scipy.optimize.minimize.Projections}
 * needs: {@link #getU()}, {@link #getS()}, {@link #getV()}. Backed by
 * LAPACK {@code dgesvd}.
 *
 * <p>{@link #getS()} returns an {@code m × n} matrix with the singular
 * values on its diagonal (zero elsewhere), matching UJMP's convention.
 */
public final class SVDMatrix {

	private final DMatrix U;
	private final DMatrix S;
	private final DMatrix V;

	public SVDMatrix(Matrix a) {
		int m = (int) a.getRowCount();
		int n = (int) a.getColumnCount();
		double[] aFlat = MatrixIO.toColumnMajor(a);
		int k = Math.min(m, n);
		double[] sigma = new double[k];
		double[] uFlat = new double[m * m];
		double[] vtFlat = new double[n * n];
		intW info = new intW(0);
		double[] work = new double[1];
		LAPACK lapack = LAPACK.getInstance();
		lapack.dgesvd("A", "A", m, n, aFlat, m, sigma, uFlat, m, vtFlat, n, work, -1, info);
		int lwork = Math.max(1, (int) work[0]);
		work = new double[lwork];
		lapack.dgesvd("A", "A", m, n, aFlat, m, sigma, uFlat, m, vtFlat, n, work, lwork, info);
		if (info.val != 0) {
			throw new ArithmeticException("dgesvd failed: info=" + info.val);
		}

		DMatrix Umat = MatrixIO.fromColumnMajor(uFlat, m, m);
		DMatrix Smat = new DMatrix(m, n);
		for (int i = 0; i < k; ++i) {
			Smat.set(i, i, sigma[i]);
		}
		// V = (V^T)^T
		DMatrix Vt = MatrixIO.fromColumnMajor(vtFlat, n, n);
		DMatrix Vmat = new DMatrix(n, n);
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				Vmat.set(i, j, Vt.get(j, i));
			}
		}

		this.U = Umat;
		this.S = Smat;
		this.V = Vmat;
	}

	public Matrix getU() { return U; }
	public Matrix getS() { return S; }
	public Matrix getV() { return V; }

	/**
	 * Singular-value reciprocal as an {@code m × n} diagonal matrix
	 * (zeros below the smallest-detectable threshold are kept zero).
	 * Mirrors UJMP's {@code SVDMatrix.getreciprocalS()} which is the
	 * pseudo-inverse-of-S used in Moore-Penrose pseudo-inverse formulas.
	 */
	public Matrix getreciprocalS() {
		int m = (int) S.getRowCount();
		int n = (int) S.getColumnCount();
		int k = Math.min(m, n);
		DMatrix r = new DMatrix(n, m);
		double tol = Math.max(m, n) * Math.ulp(S.getAsDouble(0, 0));
		for (int i = 0; i < k; ++i) {
			double sv = S.getAsDouble(i, i);
			if (sv > tol) {
				r.set(i, i, 1.0 / sv);
			}
		}
		return r;
	}
}
