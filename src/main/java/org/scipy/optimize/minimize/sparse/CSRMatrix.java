package org.scipy.optimize.minimize.sparse;

import java.util.Arrays;

/**
 * Compressed Sparse Row matrix.
 *
 * Storage analogue of {@code scipy.sparse.csr_array}: three parallel primitive
 * arrays {@code (indptr, indices, data)} together with the logical shape.
 *
 * <pre>
 * indptr.length == rows + 1
 * indptr[i] .. indptr[i+1] is the half-open range of {@code indices}/{@code data}
 *     entries belonging to row i.
 * indices[k] is the column of the k-th stored entry (0 <= indices[k] < cols).
 * data[k]    is the value of the k-th stored entry.
 * </pre>
 *
 * Within a row the entries are kept sorted by column index — this matches the
 * canonical form scipy returns from {@code csr_array(...)} after
 * {@code sum_duplicates() / sort_indices()}, and several operations (e.g.
 * {@link #toDense()}, {@link #matvec(double[])}) rely on it being safe to walk
 * the row range linearly.
 *
 * Instances are immutable from the outside: the constructor copies its inputs.
 *
 * @see <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html">scipy.sparse.csr_array</a>
 */
public final class CSRMatrix {

	private final int rows;
	private final int cols;
	private final int[] indptr;
	private final int[] indices;
	private final double[] data;

	/**
	 * Construct from raw CSR triplets. Inputs are defensively copied. The arrays
	 * must satisfy:
	 *
	 * <ul>
	 *   <li>{@code indptr.length == rows + 1}, monotonically non-decreasing,
	 *       starting at 0 and ending at {@code indices.length}.</li>
	 *   <li>{@code indices.length == data.length} (the number of stored entries).</li>
	 *   <li>{@code 0 <= indices[k] < cols} for every k.</li>
	 *   <li>Within each row, {@code indices} is strictly increasing.</li>
	 * </ul>
	 *
	 * @param rows    number of rows
	 * @param cols    number of columns
	 * @param indptr  row pointer array, length {@code rows + 1}
	 * @param indices column index for each stored entry
	 * @param data    value for each stored entry
	 */
	public CSRMatrix(int rows, int cols, int[] indptr, int[] indices, double[] data) {
		if (rows < 0 || cols < 0) {
			throw new IllegalArgumentException("rows and cols must be non-negative");
		}
		if (indptr.length != rows + 1) {
			throw new IllegalArgumentException("indptr.length must be rows + 1");
		}
		if (indices.length != data.length) {
			throw new IllegalArgumentException("indices and data length mismatch");
		}
		if (indptr[0] != 0 || indptr[rows] != indices.length) {
			throw new IllegalArgumentException("indptr[0] must be 0 and indptr[rows] must equal nnz");
		}
		for (int i = 0; i < rows; ++i) {
			if (indptr[i + 1] < indptr[i]) {
				throw new IllegalArgumentException("indptr must be non-decreasing");
			}
			int prev = -1;
			for (int k = indptr[i]; k < indptr[i + 1]; ++k) {
				int j = indices[k];
				if (j < 0 || j >= cols) {
					throw new IllegalArgumentException("column index out of range at k=" + k);
				}
				if (j <= prev) {
					throw new IllegalArgumentException("column indices must be strictly increasing within a row");
				}
				prev = j;
			}
		}
		this.rows = rows;
		this.cols = cols;
		this.indptr = indptr.clone();
		this.indices = indices.clone();
		this.data = data.clone();
	}

	public int rows() { return rows; }
	public int cols() { return cols; }
	public int nnz() { return data.length; }

	/** @return a defensive copy of the row pointer array. */
	public int[] indptr() { return indptr.clone(); }
	/** @return a defensive copy of the column index array. */
	public int[] indices() { return indices.clone(); }
	/** @return a defensive copy of the value array. */
	public double[] data() { return data.clone(); }

	int[] indptrRef() { return indptr; }
	int[] indicesRef() { return indices; }
	double[] dataRef() { return data; }

	/**
	 * Build a CSR matrix from a dense 2D array. Zeros are dropped. The input
	 * array must be rectangular ({@code dense[i].length == cols} for all i).
	 *
	 * @param dense [m][n] row-major dense values
	 * @return CSR matrix
	 */
	/**
	 * Build from any {@link org.scipy.optimize.minimize.matrix.Matrix}. Routes
	 * sparse {@link org.scipy.optimize.minimize.matrix.SparseMatrix} sources
	 * directly to {@link org.scipy.optimize.minimize.matrix.SparseMatrix#toCSR}
	 * (no dense materialisation); dense sources go through {@link #fromDense(double[][])}.
	 */
	public static CSRMatrix fromMatrix(org.scipy.optimize.minimize.matrix.Matrix m) {
		int rows = (int) m.getRowCount();
		int cols = (int) m.getColumnCount();
		if (rows == 0 || cols == 0) {
			return SparseAssembly.zeros(rows, cols);
		}
		if (m instanceof org.scipy.optimize.minimize.matrix.SparseMatrix) {
			return ((org.scipy.optimize.minimize.matrix.SparseMatrix) m).toCSR();
		}
		return fromDense(m.toDoubleArray());
	}

	public static CSRMatrix fromDense(double[][] dense) {
		int m = dense.length;
		int n = m == 0 ? 0 : dense[0].length;
		int nnz = 0;
		for (int i = 0; i < m; ++i) {
			if (dense[i].length != n) {
				throw new IllegalArgumentException("dense input must be rectangular");
			}
			for (int j = 0; j < n; ++j) {
				if (dense[i][j] != 0.0) ++nnz;
			}
		}
		int[] indptr = new int[m + 1];
		int[] indices = new int[nnz];
		double[] data = new double[nnz];
		int k = 0;
		for (int i = 0; i < m; ++i) {
			indptr[i] = k;
			for (int j = 0; j < n; ++j) {
				double v = dense[i][j];
				if (v != 0.0) {
					indices[k] = j;
					data[k] = v;
					++k;
				}
			}
		}
		indptr[m] = k;
		return new CSRMatrix(m, n, indptr, indices, data);
	}

	/**
	 * Materialize as a dense {@code [rows][cols]} array. Newly allocated each
	 * call.
	 */
	public double[][] toDense() {
		double[][] out = new double[rows][cols];
		for (int i = 0; i < rows; ++i) {
			for (int k = indptr[i]; k < indptr[i + 1]; ++k) {
				out[i][indices[k]] = data[k];
			}
		}
		return out;
	}

	/**
	 * {@code n x n} identity in CSR.
	 *
	 * @see <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.eye_array.html">scipy.sparse.eye_array</a>
	 */
	public static CSRMatrix eye(int n) {
		if (n < 0) {
			throw new IllegalArgumentException("n must be non-negative");
		}
		int[] indptr = new int[n + 1];
		int[] indices = new int[n];
		double[] data = new double[n];
		for (int i = 0; i < n; ++i) {
			indptr[i] = i;
			indices[i] = i;
			data[i] = 1.0;
		}
		indptr[n] = n;
		return new CSRMatrix(n, n, indptr, indices, data);
	}

	/**
	 * Matrix-vector product {@code y = A x}.
	 *
	 * @param x input vector of length {@link #cols()}
	 * @return result vector of length {@link #rows()}
	 */
	public double[] matvec(double[] x) {
		if (x.length != cols) {
			throw new IllegalArgumentException("x.length must equal cols");
		}
		double[] y = new double[rows];
		for (int i = 0; i < rows; ++i) {
			double s = 0.0;
			for (int k = indptr[i]; k < indptr[i + 1]; ++k) {
				s += data[k] * x[indices[k]];
			}
			y[i] = s;
		}
		return y;
	}

	/**
	 * Transposed matrix-vector product {@code y = A^T x}.
	 *
	 * @param x input vector of length {@link #rows()}
	 * @return result vector of length {@link #cols()}
	 */
	public double[] rmatvec(double[] x) {
		if (x.length != rows) {
			throw new IllegalArgumentException("x.length must equal rows");
		}
		double[] y = new double[cols];
		for (int i = 0; i < rows; ++i) {
			double xi = x[i];
			for (int k = indptr[i]; k < indptr[i + 1]; ++k) {
				y[indices[k]] += data[k] * xi;
			}
		}
		return y;
	}

	/** Element-wise scalar multiply. Returns a new matrix. */
	public CSRMatrix multiply(double s) {
		double[] newData = new double[data.length];
		for (int k = 0; k < data.length; ++k) {
			newData[k] = data[k] * s;
		}
		return new CSRMatrix(rows, cols, indptr.clone(), indices.clone(), newData);
	}

	/**
	 * Transpose. Returns a CSC matrix backed by the natural transpose
	 * relationship: {@code A.transpose()} has the same nnz as {@code A}, with
	 * rows and columns swapped. CSC is the natural output because a CSR row
	 * traversal of {@code A} is a CSC column traversal of {@code A^T}.
	 */
	public CSCMatrix transpose() {
		return new CSCMatrix(cols, rows, indptr.clone(), indices.clone(), data.clone());
	}

	/** Convert to CSC of the same matrix (not the transpose). */
	public CSCMatrix toCSC() {
		int[] outIndptr = new int[cols + 1];
		for (int k = 0; k < indices.length; ++k) {
			outIndptr[indices[k] + 1]++;
		}
		for (int j = 0; j < cols; ++j) {
			outIndptr[j + 1] += outIndptr[j];
		}
		int[] pos = Arrays.copyOf(outIndptr, cols + 1);
		int[] outIndices = new int[indices.length];
		double[] outData = new double[data.length];
		for (int i = 0; i < rows; ++i) {
			for (int k = indptr[i]; k < indptr[i + 1]; ++k) {
				int j = indices[k];
				int dest = pos[j];
				outIndices[dest] = i;
				outData[dest] = data[k];
				pos[j] = dest + 1;
			}
		}
		return new CSCMatrix(rows, cols, outIndptr, outIndices, outData);
	}

	/** Start an incremental DOK-style build. See {@link Builder}. */
	public static Builder builder(int rows, int cols) {
		return new Builder(rows, cols);
	}

	/**
	 * Incremental builder for {@link CSRMatrix} using a Dictionary-of-Keys
	 * (DOK) representation under the hood. Cheap {@code O(1)} amortised
	 * {@link #set(int, int, double)}; {@link #build()} compresses to canonical
	 * CSR (column indices sorted within each row, no duplicates).
	 *
	 * <p>Storing a value of exactly {@code 0.0} via {@code set(i, j, 0.0)}
	 * removes any previously-stored entry at {@code (i, j)} so the resulting
	 * CSR contains no explicit zeros.
	 */
	public static final class Builder {

		private final int rows;
		private final int cols;
		private final java.util.HashMap<Long, Double> entries = new java.util.HashMap<>();

		private Builder(int rows, int cols) {
			if (rows < 0 || cols < 0) {
				throw new IllegalArgumentException("rows and cols must be non-negative");
			}
			this.rows = rows;
			this.cols = cols;
		}

		public int rows() { return rows; }
		public int cols() { return cols; }

		private long key(int i, int j) {
			if (i < 0 || i >= rows || j < 0 || j >= cols) {
				throw new IndexOutOfBoundsException("(" + i + ", " + j + ") not in "
						+ rows + "x" + cols);
			}
			return ((long) i) * cols + j;
		}

		/**
		 * Store {@code v} at {@code (i, j)}. Setting to {@code 0.0} removes
		 * the entry. Repeated calls overwrite (this is set, not add).
		 */
		public Builder set(int i, int j, double v) {
			long k = key(i, j);
			if (v == 0.0) {
				entries.remove(k);
			} else {
				entries.put(k, v);
			}
			return this;
		}

		/** Read back the value at {@code (i, j)}; defaults to {@code 0.0}. */
		public double get(int i, int j) {
			Double v = entries.get(key(i, j));
			return v == null ? 0.0 : v;
		}

		public int nnz() { return entries.size(); }

		/** Compress the DOK contents into a canonical CSR representation. */
		public CSRMatrix build() {
			int nnz = entries.size();
			int[] indptr = new int[rows + 1];
			int[] indices = new int[nnz];
			double[] data = new double[nnz];

			// Bucket-count entries per row.
			int[] rowCount = new int[rows];
			for (Long k : entries.keySet()) {
				int i = (int) (k / cols);
				++rowCount[i];
			}
			indptr[0] = 0;
			for (int i = 0; i < rows; ++i) {
				indptr[i + 1] = indptr[i] + rowCount[i];
			}

			// Place entries; cursor per row = indptr[i] + offset.
			int[] cursor = indptr.clone();
			for (java.util.Map.Entry<Long, Double> e : entries.entrySet()) {
				long k = e.getKey();
				int i = (int) (k / cols);
				int j = (int) (k - (long) i * cols);
				int dest = cursor[i]++;
				indices[dest] = j;
				data[dest] = e.getValue();
			}

			// Sort each row's slice by column index (canonical CSR).
			for (int i = 0; i < rows; ++i) {
				int from = indptr[i];
				int to = indptr[i + 1];
				if (to - from > 1) {
					sortRowSlice(indices, data, from, to);
				}
			}

			return new CSRMatrix(rows, cols, indptr, indices, data);
		}

		private static void sortRowSlice(int[] indices, double[] data, int from, int to) {
			// Insertion sort (rows are typically short; avoids paired-array boxing).
			for (int k = from + 1; k < to; ++k) {
				int idx = indices[k];
				double val = data[k];
				int p = k - 1;
				while (p >= from && indices[p] > idx) {
					indices[p + 1] = indices[p];
					data[p + 1] = data[p];
					--p;
				}
				indices[p + 1] = idx;
				data[p + 1] = val;
			}
		}
	}
}
