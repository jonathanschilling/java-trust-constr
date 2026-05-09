package org.scipy.optimize.minimize.matrix;

/**
 * Forces a result to be dense regardless of the input matrix type — used by
 * callers that have a possibly-sparse matrix but want a guaranteed dense
 * representation downstream (e.g. {@link org.scipy.optimize.minimize.VectorFunction}
 * after a finite-difference Jacobian build).
 */
public final class DenseMatrix {

	private DenseMatrix() {}

	public static final class Factory {

		private Factory() {}

		public static Matrix copyFromMatrix(Matrix m) {
			int rows = (int) m.getRowCount();
			int cols = (int) m.getColumnCount();
			DMatrix out = new DMatrix(rows, cols);
			for (int i = 0; i < rows; ++i) {
				for (int j = 0; j < cols; ++j) {
					out.set(i, j, m.getAsDouble(i, j));
				}
			}
			return out;
		}
	}
}
