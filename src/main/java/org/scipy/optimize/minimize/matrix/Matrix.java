package org.scipy.optimize.minimize.matrix;

import java.util.Arrays;

/**
 * Abstract dense/sparse matrix base. Two concrete subclasses:
 * <ul>
 *   <li>{@link DenseMatrix} -- column-major {@code double[]} buffer; the BLAS
 *       and LAPACK kernels in {@link LinAlg} take {@link DenseMatrix#data()}
 *       directly with no copy.</li>
 *   <li>{@link SparseMatrix} -- Dictionary-of-Keys; cheap incremental
 *       {@link #setAsDouble} plus
 *       {@link SparseMatrix#toCSR()} to materialise a
 *       {@link org.scipy.optimize.minimize.sparse.CSRMatrix} for compute.</li>
 * </ul>
 *
 * <p>The base class exposes only the operations both layouts must support:
 * shape queries, element get/set, arithmetic, norms, slicing, scalar
 * extraction, and an in-place {@code fill}/{@code absInPlace}. The fast
 * paths (BLAS dgemm / dsyr / dsyr2, LU/QR/SVD/Cholesky via LAPACK) live on
 * {@link DenseMatrix} and {@link LinAlg}.
 */
public abstract class Matrix {

	/** @return number of rows */
	public abstract long getRowCount();
	/** @return number of columns */
	public abstract long getColumnCount();

	/** @return shape as a length-2 array {@code [rows, cols]} */
	public long[] getSize() {
		return new long[] {getRowCount(), getColumnCount()};
	}

	/**
	 * @param row 0-based row index
	 * @param col 0-based column index
	 * @return entry at {@code (row, col)}
	 */
	public abstract double getAsDouble(long row, long col);
	/**
	 * @param v   value to store
	 * @param row 0-based row index
	 * @param col 0-based column index
	 */
	public abstract void setAsDouble(double v, long row, long col);

	/**
	 * Solve {@code this * x = rhs} via LAPACK {@code dgesv} (general LU).
	 * Routes through {@link LinAlg#solve(DenseMatrix, DenseMatrix)} after
	 * densifying both operands.
	 *
	 * @param rhs right-hand side
	 * @return solution {@code x}
	 */
	public Matrix solve(Matrix rhs) {
		return LinAlg.solve(DenseMatrix.copyFromMatrix(this), DenseMatrix.copyFromMatrix(rhs));
	}

	/** @return {@code true} if this is a sparse matrix */
	public boolean isSparse() {
		return false;
	}

	/** @return a fresh row-major {@code double[][]} copy of the contents */
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

	/** @return a deep copy of this matrix */
	public abstract Matrix copy();

	/**
	 * @return contents of an {@code n x 1} column vector as a fresh
	 *         {@code double[n]}; throws if the matrix is not a column vector
	 */
	public double[] toColumnArray() {
		int n = (int) getRowCount();
		double[] out = new double[n];
		for (int i = 0; i < n; ++i) {
			out[i] = getAsDouble(i, 0);
		}
		return out;
	}

	/**
	 * @param other right operand
	 * @return matrix product {@code this * other}
	 */
	public abstract Matrix mtimes(Matrix other);

	/**
	 * @param other right operand
	 * @return element-wise sum {@code this + other}
	 */
	public abstract Matrix plus(Matrix other);

	/**
	 * @param other right operand
	 * @return element-wise difference {@code this - other}
	 */
	public abstract Matrix minus(Matrix other);

	/**
	 * @param scalar value to add elementwise
	 * @return {@code this + scalar}
	 */
	public Matrix plus(double scalar) {
		return mapAdd(scalar);
	}

	/**
	 * @param scalar value to subtract elementwise
	 * @return {@code this - scalar}
	 */
	public Matrix minus(double scalar) {
		return mapAdd(-scalar);
	}

	/**
	 * @param alpha scalar
	 * @return {@code alpha * this}
	 */
	public abstract Matrix times(double alpha);

	/**
	 * @param elementwise same-shape factor
	 * @return Hadamard product {@code this .* elementwise}
	 */
	public abstract Matrix times(Matrix elementwise);

	/**
	 * @param alpha scalar
	 * @return {@code this / alpha}
	 */
	public abstract Matrix divide(double alpha);

	/**
	 * @param elementwise same-shape divisor
	 * @return Hadamard division {@code this ./ elementwise}
	 */
	public abstract Matrix divide(Matrix elementwise);
	/** @return the transpose of this matrix */
	public abstract Matrix transpose();
	/** @return element-wise absolute value as a fresh matrix */
	public abstract Matrix abs();

	/** @return Frobenius norm */
	public abstract double norm2();
	/** @return infinity-norm (largest absolute entry) */
	public abstract double normInf();

	/**
	 * Slice rows {@code [r0..r1]} and columns {@code [c0..c1]} (inclusive).
	 * Always returns a fresh dense matrix.
	 *
	 * @param r0 first row (inclusive)
	 * @param c0 first column (inclusive)
	 * @param r1 last row (inclusive)
	 * @param c1 last column (inclusive)
	 * @return the {@code (r1-r0+1) x (c1-c0+1)} slice
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

	/**
	 * @param col 0-based column index
	 * @return that single column as an {@code n x 1} matrix
	 */
	public Matrix selectColumns(long col) {
		int rows = (int) getRowCount();
		DenseMatrix out = DenseMatrix.zeros(rows, 1);
		for (int i = 0; i < rows; ++i) {
			out.set(i, 0, getAsDouble(i, col));
		}
		return out;
	}

	/**
	 * In-place absolute value.
	 *
	 * @return {@code this} after replacing every entry by its absolute value
	 */
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

	/**
	 * Rank computed via SVD: count singular values strictly greater than the
	 * default LAPACK numerical-zero threshold {@code max(m,n) * ulp(sigma_max)}.
	 *
	 * @return numerical rank (zero for empty matrices)
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
	 * Lenient shape predicate: returns {@code true} if either dimension is 1 --
	 * i.e. the matrix is shaped like a vector (row, column, or 1&times;1 scalar).
	 * Used by BFGS / SR1 / FullHessianUpdateStrategy as a shape precondition
	 * on their step / gradient inputs.
	 *
	 * @return {@code true} if either {@code rows == 1} or {@code cols == 1}
	 */
	public boolean isRowVector() {
		return getRowCount() == 1 || getColumnCount() == 1;
	}

	/** @return {@code true} iff this matrix has exactly one column */
	public boolean isColumnVector() {
		return getColumnCount() == 1;
	}

	/** @return the Frobenius norm; alias for {@link #norm2()} kept for clarity */
	public double normF() {
		return norm2();
	}

	/**
	 * @param other matrix to compare against
	 * @return {@code true} iff {@code other} has the same shape and every
	 *         entry compares bit-equal under {@code ==}
	 */
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

	/**
	 * In-place fill with the given scalar.
	 *
	 * @param v value to assign to every entry
	 * @return {@code this} after the mutation
	 */
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

	/**
	 * Extract the single scalar value of a {@code 1 x 1} matrix.
	 *
	 * @return the only entry
	 * @throws IllegalStateException if the matrix is not {@code 1 x 1}
	 */
	public double doubleValue() {
		if (getRowCount() != 1 || getColumnCount() != 1) {
			throw new IllegalStateException("doubleValue() requires 1x1 matrix; got "
					+ getRowCount() + "x" + getColumnCount());
		}
		return getAsDouble(0, 0);
	}

	/**
	 * Helper used by {@link #plus(double)} and {@link #minus(double)}: returns
	 * a fresh dense matrix with {@code scalar} added to every entry.
	 *
	 * @param scalar value to add elementwise
	 * @return a fresh {@link DenseMatrix} of the same shape
	 */
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
	 * (column-major {@code double[]}) -- kept under {@code Matrix.Factory.*} so
	 * existing call sites compile unchanged during the staged migration.
	 */
	public static final class Factory {

		private Factory() {}

		/**
		 * @param rows number of rows
		 * @param cols number of columns
		 * @return a fresh {@code rows x cols} zero matrix
		 */
		public static DenseMatrix zeros(long rows, long cols) {
			return DenseMatrix.zeros((int) rows, (int) cols);
		}

		/**
		 * @param size length-2 shape array {@code [rows, cols]} (length-1
		 *             interpreted as a column vector)
		 * @return a fresh zero matrix of the requested shape
		 */
		public static DenseMatrix zeros(long[] size) {
			long r = size.length > 0 ? size[0] : 0;
			long c = size.length > 1 ? size[1] : 1;
			return zeros(r, c);
		}

		/**
		 * @param rows number of rows
		 * @param cols number of columns
		 * @return a fresh {@code rows x cols} identity-like matrix
		 *         ({@code 1.0} on the leading diagonal, {@code 0.0} elsewhere)
		 */
		public static DenseMatrix eye(long rows, long cols) {
			int r = (int) rows;
			int c = (int) cols;
			DenseMatrix m = DenseMatrix.zeros(r, c);
			int n = Math.min(r, c);
			for (int i = 0; i < n; ++i) m.set(i, i, 1.0);
			return m;
		}

		/**
		 * @param rows number of rows
		 * @param cols number of columns
		 * @return a fresh matrix filled with {@code 1.0}
		 */
		public static DenseMatrix ones(long rows, long cols) {
			return fill(1.0, rows, cols);
		}

		/**
		 * @param size length-2 shape array {@code [rows, cols]}
		 * @return a fresh matrix filled with {@code 1.0}
		 */
		public static DenseMatrix ones(long[] size) {
			long r = size.length > 0 ? size[0] : 0;
			long c = size.length > 1 ? size[1] : 1;
			return ones(r, c);
		}

		/**
		 * @param v    value to fill every entry with
		 * @param rows number of rows
		 * @param cols number of columns
		 * @return a fresh {@code rows x cols} matrix with all entries equal to {@code v}
		 */
		public static DenseMatrix fill(double v, long rows, long cols) {
			DenseMatrix m = DenseMatrix.zeros((int) rows, (int) cols);
			Arrays.fill(m.data(), v);
			return m;
		}

		/**
		 * @param arr row-major source ({@code arr[i][j]} = entry at {@code (i, j)})
		 * @return a fresh dense matrix with the same shape and content
		 */
		public static DenseMatrix linkToArray(double[][] arr) {
			return DenseMatrix.fromRows(arr);
		}

		/**
		 * Wrap a 1-D array as an {@code n x 1} column vector. For example,
		 * {@code linkToArray(new double[] {a, b})} produces a 2&times;1 vector
		 * indexed via {@code getAsDouble(i, 0)}.
		 *
		 * @param colVec entries of the resulting column vector
		 * @return an {@code n x 1} column vector
		 */
		public static DenseMatrix linkToArray(double[] colVec) {
			return DenseMatrix.column(colVec);
		}

		/**
		 * Alias for {@link #linkToArray(double[][])}.
		 *
		 * @param arr row-major source
		 * @return a fresh dense matrix
		 */
		public static DenseMatrix importFromArray(double[][] arr) {
			return DenseMatrix.fromRows(arr);
		}

		/**
		 * Wrap a 1-D array as a {@code 1 x n} row matrix (the row-vector
		 * counterpart of {@link #linkToArray(double[])}). Test fixtures that
		 * need a column then call {@code .transpose()}.
		 *
		 * @param rowVec entries of the resulting row vector
		 * @return a {@code 1 x n} row vector
		 */
		public static DenseMatrix importFromArray(double[] rowVec) {
			return DenseMatrix.row(rowVec);
		}

		/**
		 * {@code int[][]} input -- promoted to {@code double} for test fixtures.
		 *
		 * @param arr row-major integer source
		 * @return a fresh dense matrix with the same shape and content
		 */
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
		 * Random normal matrix with mean 0 and stddev 1. The underlying
		 * {@link java.util.Random} is seedable via {@link #setRandSeed(long)}.
		 *
		 * @param rows number of rows
		 * @param cols number of columns
		 * @return a fresh {@code rows x cols} matrix populated with i.i.d.
		 *         standard-normal samples
		 */
		public static DenseMatrix randn(long rows, long cols) {
			DenseMatrix m = DenseMatrix.zeros((int) rows, (int) cols);
			double[] d = m.data();
			for (int k = 0; k < d.length; ++k) d[k] = RNG.nextGaussian();
			return m;
		}

		/**
		 * Set the seed of the RNG underlying {@link #randn(long, long)} so
		 * results become reproducible.
		 *
		 * @param seed RNG seed
		 */
		public static void setRandSeed(long seed) {
			RNG.setSeed(seed);
		}

		private static final java.util.Random RNG = new java.util.Random();

		/**
		 * @param m source matrix
		 * @return a deep copy of {@code m}
		 */
		public static Matrix copyFromMatrix(Matrix m) {
			return m.copy();
		}

		/**
		 * Vertical concatenation: stack the input matrices on top of each
		 * other. All inputs must have the same column count.
		 *
		 * @param parts matrices to stack (in order, from top to bottom)
		 * @return a fresh {@link DenseMatrix} containing the stacked rows
		 * @throws IllegalArgumentException if column counts differ
		 */
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
