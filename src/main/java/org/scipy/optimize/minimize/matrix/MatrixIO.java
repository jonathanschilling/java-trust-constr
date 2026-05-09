package org.scipy.optimize.minimize.matrix;

/**
 * Helpers for converting between {@link Matrix} (row-major Java view) and
 * the column-major {@code double[]} layout LAPACK expects.
 */
final class MatrixIO {

	private MatrixIO() {}

	static double[] toColumnMajor(Matrix m) {
		int rows = (int) m.getRowCount();
		int cols = (int) m.getColumnCount();
		double[] out = new double[rows * cols];
		for (int j = 0; j < cols; ++j) {
			for (int i = 0; i < rows; ++i) {
				out[i + j * rows] = m.getAsDouble(i, j);
			}
		}
		return out;
	}

	static DMatrix fromColumnMajor(double[] flat, int rows, int cols) {
		DMatrix out = new DMatrix(rows, cols);
		for (int j = 0; j < cols; ++j) {
			for (int i = 0; i < rows; ++i) {
				out.set(i, j, flat[i + j * rows]);
			}
		}
		return out;
	}
}
