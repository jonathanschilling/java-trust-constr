package org.scipy.optimize.minimize.matrix;

/**
 * Small static helpers that operate on {@code double[]} vectors and
 * {@link Matrix} instances. Replaces the previously-vendored {@code LinAlg}
 * adapter which lived at the top level of the trust-constr port.
 */
public final class MatrixOps {

	private MatrixOps() {}

	/** Euclidean (2-)norm of a vector. */
	public static double norm2(double[] v) {
		double sum = 0.0;
		for (int i = 0; i < v.length; ++i) {
			sum += v[i] * v[i];
		}
		return Math.sqrt(sum);
	}

	/** Dot product of two equal-length vectors. */
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

	/** Diagonal matrix with the given entries on the diagonal (sparse). */
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
	 */
	public static Matrix sparse(Matrix A) {
		Matrix sparseA = SparseMatrix.Factory.zeros(A.getRowCount(), A.getColumnCount());
		for (long[] pos : A.allCoordinates()) {
			double v = A.getAsDouble(pos);
			if (v != 0.0) {
				sparseA.setAsDouble(v, pos);
			}
		}
		return sparseA;
	}
}
