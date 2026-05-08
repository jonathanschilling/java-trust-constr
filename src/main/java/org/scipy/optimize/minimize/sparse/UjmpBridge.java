package org.scipy.optimize.minimize.sparse;

import org.ujmp.core.Matrix;

/**
 * Conversions between UJMP {@link Matrix} (the legacy in-tree representation)
 * and the {@link CSRMatrix} / {@link CSCMatrix} types of this package.
 *
 * <p>This bridge exists because the trust-constr port is being moved off of
 * UJMP onto its own scipy-style sparse module incrementally. Algorithm classes
 * still take and return UJMP {@code Matrix} objects on their public API; the
 * new module is wired in <em>under</em> those signatures by converting at the
 * boundary. Once a class has been fully ported off UJMP its calls to this
 * bridge can be deleted.
 */
public final class UjmpBridge {

	private UjmpBridge() { }

	/**
	 * Convert a UJMP {@link Matrix} (dense or sparse) to a {@link CSRMatrix}.
	 * Materialises through {@code toDoubleArray()} first; zeros are dropped.
	 */
	public static CSRMatrix toCSR(Matrix m) {
		double[][] dense = m.toDoubleArray();
		return CSRMatrix.fromDense(dense);
	}

	/** Convert a UJMP {@link Matrix} to a {@link CSCMatrix}. */
	public static CSCMatrix toCSC(Matrix m) {
		double[][] dense = m.toDoubleArray();
		return CSCMatrix.fromDense(dense);
	}

	/**
	 * Materialise a CSR matrix to a UJMP dense {@link Matrix}.
	 */
	public static Matrix toUjmp(CSRMatrix a) {
		return Matrix.Factory.linkToArray(a.toDense());
	}

	/** Materialise a CSC matrix to a UJMP dense {@link Matrix}. */
	public static Matrix toUjmp(CSCMatrix a) {
		return Matrix.Factory.linkToArray(a.toDense());
	}

	/**
	 * Extract a UJMP column vector ({@code n x 1}) as a {@code double[n]}.
	 * Equivalent to {@code LinAlg.col} but kept here so the sparse module is
	 * not coupled to the legacy {@code LinAlg} class.
	 */
	public static double[] colToArray(Matrix v) {
		long rows = v.getRowCount();
		double[] out = new double[(int) rows];
		for (int i = 0; i < rows; ++i) {
			out[i] = v.getAsDouble(i, 0);
		}
		return out;
	}

	/** Wrap a {@code double[n]} as a UJMP column vector ({@code n x 1}). */
	public static Matrix arrayToCol(double[] v) {
		double[][] data = new double[v.length][1];
		for (int i = 0; i < v.length; ++i) {
			data[i][0] = v[i];
		}
		return Matrix.Factory.linkToArray(data);
	}
}
