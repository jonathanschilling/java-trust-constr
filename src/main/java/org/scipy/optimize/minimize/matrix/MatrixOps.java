package org.scipy.optimize.minimize.matrix;

/**
 * Small static helpers that operate on {@code double[]} vectors and
 * {@link Matrix} instances.
 */
public final class MatrixOps {

	private MatrixOps() {}

	/**
	 * Euclidean (2-)norm of a vector.
	 *
	 * @param v vector
	 * @return {@code sqrt(sum_i v[i]^2)}
	 */
	public static double norm2(double[] v) {
		double sum = 0.0;
		for (int i = 0; i < v.length; ++i) {
			sum += v[i] * v[i];
		}
		return Math.sqrt(sum);
	}

	/**
	 * Dot product of two equal-length vectors.
	 *
	 * @param a left operand
	 * @param b right operand (must have the same length as {@code a})
	 * @return {@code sum_i a[i] * b[i]}
	 */
	public static double dot(double[] a, double[] b) {
		if (a.length != b.length) {
			throw new IllegalArgumentException("dot(): length mismatch "
					+ a.length + " vs " + b.length);
		}
		double s = 0.0;
		for (int i = 0; i < a.length; ++i) {
			s += a[i] * b[i];
		}
		return s;
	}

	/**
	 * Build a sparse diagonal matrix with the given entries on the diagonal.
	 *
	 * @param diagonal length-{@code n} array of diagonal values
	 * @return {@code n x n} {@link SparseMatrix} with the diagonal populated
	 *         and zeros elsewhere
	 */
	public static Matrix diag(double[] diagonal) {
		int n = diagonal.length;
		SparseMatrix d = SparseMatrix.Factory.zeros(n, n);
		for (int i = 0; i < n; ++i) {
			d.setAsDouble(diagonal[i], i, i);
		}
		return d;
	}

	/**
	 * Materialise the non-zero entries of a possibly-dense matrix into a
	 * {@link SparseMatrix} (DOK-backed). Used by tests that exercise the
	 * sparse code paths against a dense reference.
	 *
	 * @param A source matrix
	 * @return {@link SparseMatrix} with the same shape and non-zero entries
	 */
	public static Matrix sparse(Matrix A) {
		int rows = (int) A.getRowCount();
		int cols = (int) A.getColumnCount();
		Matrix sparseA = SparseMatrix.Factory.zeros(rows, cols);
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				double v = A.getAsDouble(i, j);
				if (v != 0.0) {
					sparseA.setAsDouble(v, i, j);
				}
			}
		}
		return sparseA;
	}
}
