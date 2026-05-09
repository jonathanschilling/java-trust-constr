package org.scipy.optimize.minimize.matrix;

/**
 * Abstract dense/sparse matrix base. Two concrete subclasses:
 * <ul>
 *   <li>{@link DenseMatrix} — column-major {@code double[]} buffer; the BLAS
 *       and LAPACK kernels in {@link LinAlg} take {@link DenseMatrix#data()}
 *       directly with no copy.</li>
 *   <li>{@link SparseMatrix} — Dictionary-of-Keys; cheap incremental
 *       {@link #setAsDouble} plus
 *       {@link SparseMatrix#toCSR()} to materialise a
 *       {@link org.scipy.optimize.minimize.sparse.CSRMatrix} for compute.</li>
 * </ul>
 *
 * <p>The base class exposes the operations both layouts must support
 * (shape queries, element get/set, arithmetic, norms, slicing, simple
 * boolean comparisons used by {@link org.scipy.optimize.minimize.NumDiff}).
 * Specialised paths (BLAS dgemm / dsyr / dsyr2, LU/QR/SVD/Cholesky via
 * LAPACK) live on {@link DenseMatrix} and {@link LinAlg}.
 */
public abstract class Matrix {

	public abstract long getRowCount();
	public abstract long getColumnCount();

	public long[] getSize() {
		return new long[] {getRowCount(), getColumnCount()};
	}

	public abstract double getAsDouble(long row, long col);
	public abstract void setAsDouble(double v, long row, long col);

	/** UJMP-compatible variadic position accessors. */
	public double getAsDouble(long... pos) {
		return getAsDouble(pos[0], pos[1]);
	}
	public void setAsDouble(double v, long... pos) {
		setAsDouble(v, pos[0], pos[1]);
	}

	/**
	 * Iterate over every {@code (row, col)} index of this matrix in row-major
	 * order. Returned arrays are reused across iterations (consumers must copy
	 * if they need to retain values).
	 */
	public Iterable<long[]> allCoordinates() {
		final long rows = getRowCount();
		final long cols = getColumnCount();
		return () -> new java.util.Iterator<long[]>() {
			private long r = 0;
			private long c = 0;
			private final long[] pos = new long[2];

			@Override
			public boolean hasNext() {
				return r < rows && cols > 0;
			}

			@Override
			public long[] next() {
				pos[0] = r;
				pos[1] = c;
				++c;
				if (c >= cols) {
					c = 0;
					++r;
				}
				return pos;
			}
		};
	}

	/**
	 * Solve {@code this * x = rhs} via LAPACK {@code dgesv} (general LU).
	 * Routes through {@link LinAlg#solve(DenseMatrix, DenseMatrix)} after
	 * densifying both operands.
	 */
	public Matrix solve(Matrix rhs) {
		return LinAlg.solve(DenseMatrix.copyFromMatrix(this), DenseMatrix.copyFromMatrix(rhs));
	}

	public boolean isSparse() {
		return false;
	}

	public double[][] toDoubleArray() {
		int n = (int) getRowCount();
		int m = (int) getColumnCount();
		double[][] out = new double[n][m];
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				out[i][j] = getAsDouble(i, j);
			}
		}
		return out;
	}

	public abstract Matrix copy();

	/**
	 * Column vector ({@code n × 1}) extracted as a flat {@code double[n]}.
	 * Replaces the previously-vendored {@code ....toColumnArray()} helper.
	 */
	public double[] toColumnArray() {
		int n = (int) getRowCount();
		double[] out = new double[n];
		for (int i = 0; i < n; ++i) {
			out[i] = getAsDouble(i, 0);
		}
		return out;
	}

	public abstract Matrix mtimes(Matrix other);
	public abstract Matrix plus(Matrix other);
	public abstract Matrix minus(Matrix other);
	public Matrix plus(double scalar) {
		return mapAdd(scalar);
	}
	public Matrix minus(double scalar) {
		return mapAdd(-scalar);
	}
	public abstract Matrix times(double alpha);
	public abstract Matrix times(Matrix elementwise);
	public abstract Matrix divide(double alpha);
	public abstract Matrix divide(Matrix elementwise);
	public abstract Matrix transpose();
	public abstract Matrix abs();

	public abstract double norm2();
	public abstract double normInf();

	/**
	 * Slice rows {@code [r0..r1]} and columns {@code [c0..c1]} (inclusive).
	 * Always returns a fresh dense matrix.
	 */
	public Matrix subMatrix(long r0, long c0, long r1, long c1) {
		int rows = (int) (r1 - r0 + 1);
		int cols = (int) (c1 - c0 + 1);
		DenseMatrix out = DenseMatrix.zeros(rows, cols);
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				out.set(i, j, getAsDouble(r0 + i, c0 + j));
			}
		}
		return out;
	}

	/** Select a single column. */
	public Matrix selectColumns(long col) {
		int rows = (int) getRowCount();
		DenseMatrix out = DenseMatrix.zeros(rows, 1);
		for (int i = 0; i < rows; ++i) {
			out.set(i, 0, getAsDouble(i, col));
		}
		return out;
	}

	/** In-place absolute value; returns {@code this} after mutation. */
	public Matrix absInPlace() {
		int rows = (int) getRowCount();
		int cols = (int) getColumnCount();
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				double v = getAsDouble(i, j);
				if (v < 0.0) setAsDouble(-v, i, j);
			}
		}
		return this;
	}

	/** Element-wise less-than. Result entries are 1.0 / 0.0. */
	public Matrix lt(Matrix other) {
		return elementwiseCompare(other, (a, b) -> a < b);
	}
	public Matrix lt(double scalar) {
		return elementwiseCompareScalar(scalar, (a, b) -> a < b);
	}
	public Matrix gt(Matrix other) {
		return elementwiseCompare(other, (a, b) -> a > b);
	}
	public Matrix gt(double scalar) {
		return elementwiseCompareScalar(scalar, (a, b) -> a > b);
	}
	public Matrix le(Matrix other) {
		return elementwiseCompare(other, (a, b) -> a <= b);
	}
	public Matrix le(double scalar) {
		return elementwiseCompareScalar(scalar, (a, b) -> a <= b);
	}
	public Matrix ge(Matrix other) {
		return elementwiseCompare(other, (a, b) -> a >= b);
	}
	public Matrix ge(double scalar) {
		return elementwiseCompareScalar(scalar, (a, b) -> a >= b);
	}
	public Matrix eq(Matrix other) {
		return elementwiseCompare(other, (a, b) -> a == b);
	}

	public Matrix not() {
		int rows = (int) getRowCount();
		int cols = (int) getColumnCount();
		DenseMatrix out = DenseMatrix.zeros(rows, cols);
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				out.set(i, j, getAsDouble(i, j) == 0.0 ? 1.0 : 0.0);
			}
		}
		return out;
	}
	public Matrix and(Matrix other) {
		return elementwiseCompare(other, (a, b) -> a != 0.0 && b != 0.0);
	}
	public Matrix or(Matrix other) {
		return elementwiseCompare(other, (a, b) -> a != 0.0 || b != 0.0);
	}

	/** Cast booleans-as-doubles back to integer-valued doubles (no-op here). */
	public Matrix toIntMatrix() {
		int rows = (int) getRowCount();
		int cols = (int) getColumnCount();
		DenseMatrix out = DenseMatrix.zeros(rows, cols);
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				out.set(i, j, (double) (long) getAsDouble(i, j));
			}
		}
		return out;
	}

	/**
	 * Rank computed via SVD: count singular values strictly greater than the
	 * default LAPACK numerical-zero threshold {@code max(m,n) * ulp(σ_max)}.
	 */
	public long rank() {
		int m = (int) getRowCount();
		int n = (int) getColumnCount();
		int k = Math.min(m, n);
		if (k == 0) return 0;
		SVDResult svd = LinAlg.svd(DenseMatrix.copyFromMatrix(this));
		double sigmaMax = svd.s()[0];
		double tol = Math.max(m, n) * Math.ulp(sigmaMax);
		return svd.rank(tol);
	}

	/**
	 * UJMP-compat semantics: returns true if either dimension is 1 (i.e., the
	 * matrix is shaped like a vector — row, column, or scalar). This matches
	 * the lenient {@code isRowVector()} used by the trust-constr port for its
	 * shape preconditions.
	 */
	public boolean isRowVector() {
		return getRowCount() == 1 || getColumnCount() == 1;
	}

	public boolean isColumnVector() {
		return getColumnCount() == 1;
	}

	public double normF() {
		return norm2();
	}

	public boolean equalsContent(Matrix other) {
		if (other == null) return false;
		if (other.getRowCount() != getRowCount() || other.getColumnCount() != getColumnCount()) {
			return false;
		}
		int rows = (int) getRowCount();
		int cols = (int) getColumnCount();
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				if (getAsDouble(i, j) != other.getAsDouble(i, j)) return false;
			}
		}
		return true;
	}

	/** In-place fill with {@code v}; returns {@code this} after mutation. */
	public Matrix fill(double v) {
		int rows = (int) getRowCount();
		int cols = (int) getColumnCount();
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				setAsDouble(v, i, j);
			}
		}
		return this;
	}

	public boolean getAsBoolean(long row, long col) {
		return getAsDouble(row, col) != 0.0;
	}
	public boolean getAsBoolean(long... pos) {
		return getAsDouble(pos[0], pos[1]) != 0.0;
	}
	public void setAsBoolean(boolean v, long row, long col) {
		setAsDouble(v ? 1.0 : 0.0, row, col);
	}
	public void setAsBoolean(boolean v, long... pos) {
		setAsDouble(v ? 1.0 : 0.0, pos[0], pos[1]);
	}

	public int getAsInt(long row, long col) {
		return (int) getAsDouble(row, col);
	}
	public int getAsInt(long... pos) {
		return (int) getAsDouble(pos[0], pos[1]);
	}
	public void setAsInt(int v, long row, long col) {
		setAsDouble(v, row, col);
	}
	public void setAsInt(int v, long... pos) {
		setAsDouble(v, pos[0], pos[1]);
	}

	/**
	 * Iterate only over the (row, col) coordinates that hold non-zero values
	 * (for sparse matrices). The default dense implementation enumerates every
	 * cell — sparse subclasses should override for efficiency.
	 */
	public Iterable<long[]> availableCoordinates() {
		return allCoordinates();
	}

	public double getValueSum() {
		int rows = (int) getRowCount();
		int cols = (int) getColumnCount();
		double sum = 0.0;
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				sum += getAsDouble(i, j);
			}
		}
		return sum;
	}

	@FunctionalInterface
	private interface DoubleBiPredicate {
		boolean test(double a, double b);
	}

	private Matrix elementwiseCompare(Matrix other, DoubleBiPredicate p) {
		int rows = (int) getRowCount();
		int cols = (int) getColumnCount();
		DenseMatrix out = DenseMatrix.zeros(rows, cols);
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				out.set(i, j, p.test(getAsDouble(i, j), other.getAsDouble(i, j)) ? 1.0 : 0.0);
			}
		}
		return out;
	}

	private Matrix elementwiseCompareScalar(double scalar, DoubleBiPredicate p) {
		int rows = (int) getRowCount();
		int cols = (int) getColumnCount();
		DenseMatrix out = DenseMatrix.zeros(rows, cols);
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				out.set(i, j, p.test(getAsDouble(i, j), scalar) ? 1.0 : 0.0);
			}
		}
		return out;
	}

	public double doubleValue() {
		if (getRowCount() != 1 || getColumnCount() != 1) {
			throw new IllegalStateException("doubleValue() requires 1x1 matrix; got "
					+ getRowCount() + "x" + getColumnCount());
		}
		return getAsDouble(0, 0);
	}

	protected Matrix mapAdd(double scalar) {
		int n = (int) getRowCount();
		int m = (int) getColumnCount();
		DenseMatrix out = DenseMatrix.zeros(n, m);
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				out.set(i, j, getAsDouble(i, j) + scalar);
			}
		}
		return out;
	}

	/**
	 * Static constructors. All paths now produce {@link DenseMatrix} instances
	 * (column-major {@code double[]}) — kept under {@code Matrix.Factory.*} so
	 * existing call sites compile unchanged during the staged migration.
	 */
	public static final class Factory {

		private Factory() {}

		public static DenseMatrix zeros(long rows, long cols) {
			return DenseMatrix.zeros((int) rows, (int) cols);
		}

		public static DenseMatrix zeros(long[] size) {
			long r = size.length > 0 ? size[0] : 0;
			long c = size.length > 1 ? size[1] : 1;
			return zeros(r, c);
		}

		public static DenseMatrix eye(long rows, long cols) {
			int r = (int) rows;
			int c = (int) cols;
			DenseMatrix m = DenseMatrix.zeros(r, c);
			int n = Math.min(r, c);
			for (int i = 0; i < n; ++i) m.set(i, i, 1.0);
			return m;
		}

		public static DenseMatrix ones(long rows, long cols) {
			return fill(1.0, rows, cols);
		}

		public static DenseMatrix ones(long[] size) {
			long r = size.length > 0 ? size[0] : 0;
			long c = size.length > 1 ? size[1] : 1;
			return ones(r, c);
		}

		public static DenseMatrix fill(double v, long rows, long cols) {
			DenseMatrix m = DenseMatrix.zeros((int) rows, (int) cols);
			java.util.Arrays.fill(m.data(), v);
			return m;
		}

		public static DenseMatrix linkToArray(double[][] arr) {
			return DenseMatrix.fromRows(arr);
		}

		/**
		 * Wrap a 1-D array as an {@code n × 1} column vector — matches the
		 * UJMP {@code Matrix.Factory.linkToArray(double[])} behaviour the port
		 * relied on (e.g. {@code linkToArray(new double[] {a, b})} produces a
		 * 2×1 vector indexed via {@code getAsDouble(i, 0)}).
		 */
		public static DenseMatrix linkToArray(double[] colVec) {
			return DenseMatrix.column(colVec);
		}

		/** UJMP-compat alias for {@link #linkToArray(double[][])}. */
		public static DenseMatrix importFromArray(double[][] arr) {
			return DenseMatrix.fromRows(arr);
		}

		/**
		 * UJMP {@code importFromArray(double[])} returns a {@code 1 × n} row
		 * matrix (independent of the n×1 convention {@link #linkToArray(double[])}
		 * uses). The trust-constr tests rely on this distinction and follow up
		 * with {@code .transpose()} when they need a column.
		 */
		public static DenseMatrix importFromArray(double[] rowVec) {
			return DenseMatrix.row(rowVec);
		}

		/** {@code int[][]} input — promoted to double for the test fixtures. */
		public static DenseMatrix linkToArray(int[][] arr) {
			int n = arr.length;
			int m = n == 0 ? 0 : arr[0].length;
			DenseMatrix out = DenseMatrix.zeros(n, m);
			for (int i = 0; i < n; ++i) {
				if (arr[i].length != m) {
					throw new IllegalArgumentException("linkToArray(int[][]): row " + i + " has length "
							+ arr[i].length + ", expected " + m);
				}
				for (int j = 0; j < m; ++j) out.set(i, j, arr[i][j]);
			}
			return out;
		}

		/**
		 * Random normal matrix with mean 0 and stddev 1. Seedable via
		 * {@link #setRandSeed(long)} (mirrors UJMP's {@code MathUtil.setSeed}).
		 */
		public static DenseMatrix randn(long rows, long cols) {
			DenseMatrix m = DenseMatrix.zeros((int) rows, (int) cols);
			double[] d = m.data();
			for (int k = 0; k < d.length; ++k) d[k] = RNG.nextGaussian();
			return m;
		}

		public static void setRandSeed(long seed) {
			RNG.setSeed(seed);
		}

		private static final java.util.Random RNG = new java.util.Random();

		public static Matrix copyFromMatrix(Matrix m) {
			return m.copy();
		}

		public static DenseMatrix vertCat(Matrix... parts) {
			if (parts.length == 0) return DenseMatrix.zeros(0, 0);
			int cols = (int) parts[0].getColumnCount();
			int totalRows = 0;
			for (Matrix p : parts) {
				if (p.getColumnCount() != cols) {
					throw new IllegalArgumentException("vertCat: column-count mismatch");
				}
				totalRows += (int) p.getRowCount();
			}
			DenseMatrix out = DenseMatrix.zeros(totalRows, cols);
			int row = 0;
			for (Matrix p : parts) {
				int pn = (int) p.getRowCount();
				for (int i = 0; i < pn; ++i) {
					for (int j = 0; j < cols; ++j) {
						out.set(row + i, j, p.getAsDouble(i, j));
					}
				}
				row += pn;
			}
			return out;
		}
	}
}
