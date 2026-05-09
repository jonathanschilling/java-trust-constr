package org.scipy.optimize.minimize.sparse;

import java.util.Arrays;

/**
 * Compressed Sparse Column matrix.
 *
 * Storage analogue of {@code scipy.sparse.csc_array}. Layout mirrors
 * {@link CSRMatrix} with rows and columns swapped:
 *
 * <pre>
 * indptr.length == cols + 1
 * indptr[j] .. indptr[j+1] is the half-open range of {@code indices}/{@code data}
 *     entries belonging to column j.
 * indices[k] is the row of the k-th stored entry (0 &le; indices[k] &lt; rows).
 * data[k]    is the value of the k-th stored entry.
 * </pre>
 *
 * Within a column the entries are kept sorted by row index.
 *
 * The CSC format is needed by trust-constr in two places:
 * <ol>
 *   <li>The KKT projection block ({@code projections.py:99}) is materialised in
 *       CSC because {@code scipy.sparse.linalg.factorized} prefers it.</li>
 *   <li>{@code A^T} of a CSR Jacobian is naturally CSC of the same data.</li>
 * </ol>
 *
 * @see <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.html">scipy.sparse.csc_array</a>
 */
public final class CSCMatrix {

	private final int rows;
	private final int cols;
	private final int[] indptr;
	private final int[] indices;
	private final double[] data;

	/**
	 * Construct from raw CSC triplets. Inputs are defensively copied. See the
	 * class-level invariants for shape and ordering requirements.
	 *
	 * @param rows    number of rows
	 * @param cols    number of columns
	 * @param indptr  column pointer array, length {@code cols + 1}
	 * @param indices row index for each stored entry
	 * @param data    value for each stored entry
	 */
	public CSCMatrix(int rows, int cols, int[] indptr, int[] indices, double[] data) {
		if (rows < 0 || cols < 0) {
			throw new IllegalArgumentException("rows and cols must be non-negative");
		}
		if (indptr.length != cols + 1) {
			throw new IllegalArgumentException("indptr.length must be cols + 1");
		}
		if (indices.length != data.length) {
			throw new IllegalArgumentException("indices and data length mismatch");
		}
		if (indptr[0] != 0 || indptr[cols] != indices.length) {
			throw new IllegalArgumentException("indptr[0] must be 0 and indptr[cols] must equal nnz");
		}
		for (int j = 0; j < cols; ++j) {
			if (indptr[j + 1] < indptr[j]) {
				throw new IllegalArgumentException("indptr must be non-decreasing");
			}
			int prev = -1;
			for (int k = indptr[j]; k < indptr[j + 1]; ++k) {
				int i = indices[k];
				if (i < 0 || i >= rows) {
					throw new IllegalArgumentException("row index out of range at k=" + k);
				}
				if (i <= prev) {
					throw new IllegalArgumentException("row indices must be strictly increasing within a column");
				}
				prev = i;
			}
		}
		this.rows = rows;
		this.cols = cols;
		this.indptr = indptr.clone();
		this.indices = indices.clone();
		this.data = data.clone();
	}

	/** @return number of rows */
	public int rows() { return rows; }
	/** @return number of columns */
	public int cols() { return cols; }
	/** @return number of stored (structural) non-zero entries */
	public int nnz() { return data.length; }

	/** @return defensive copy of the column pointer array */
	public int[] indptr() { return indptr.clone(); }
	/** @return defensive copy of the row index array */
	public int[] indices() { return indices.clone(); }
	/** @return defensive copy of the value array */
	public double[] data() { return data.clone(); }

	int[] indptrRef() { return indptr; }
	int[] indicesRef() { return indices; }
	double[] dataRef() { return data; }

	/**
	 * Build CSC from a dense 2D array; column-major scan, but the input is
	 * still given as row-major {@code dense[i][j]}.
	 *
	 * @param dense {@code [m][n]} row-major dense values
	 * @return CSC matrix with the same non-zero pattern as {@code dense}
	 */
	public static CSCMatrix fromDense(double[][] dense) {
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
		int[] indptr = new int[n + 1];
		int[] indices = new int[nnz];
		double[] data = new double[nnz];
		int k = 0;
		for (int j = 0; j < n; ++j) {
			indptr[j] = k;
			for (int i = 0; i < m; ++i) {
				double v = dense[i][j];
				if (v != 0.0) {
					indices[k] = i;
					data[k] = v;
					++k;
				}
			}
		}
		indptr[n] = k;
		return new CSCMatrix(m, n, indptr, indices, data);
	}

	/**
	 * Materialise as a dense {@code [rows][cols]} array.
	 *
	 * @return fresh row-major dense view
	 */
	public double[][] toDense() {
		double[][] out = new double[rows][cols];
		for (int j = 0; j < cols; ++j) {
			for (int k = indptr[j]; k < indptr[j + 1]; ++k) {
				out[indices[k]][j] = data[k];
			}
		}
		return out;
	}

	/**
	 * {@code n x n} identity in CSC.
	 *
	 * @param n dimension; must be non-negative
	 * @return the {@code n x n} identity matrix
	 */
	public static CSCMatrix eye(int n) {
		if (n < 0) {
			throw new IllegalArgumentException("n must be non-negative");
		}
		int[] indptr = new int[n + 1];
		int[] indices = new int[n];
		double[] data = new double[n];
		for (int j = 0; j < n; ++j) {
			indptr[j] = j;
			indices[j] = j;
			data[j] = 1.0;
		}
		indptr[n] = n;
		return new CSCMatrix(n, n, indptr, indices, data);
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
		for (int j = 0; j < cols; ++j) {
			double xj = x[j];
			if (xj == 0.0) continue;
			for (int k = indptr[j]; k < indptr[j + 1]; ++k) {
				y[indices[k]] += data[k] * xj;
			}
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
		for (int j = 0; j < cols; ++j) {
			double s = 0.0;
			for (int k = indptr[j]; k < indptr[j + 1]; ++k) {
				s += data[k] * x[indices[k]];
			}
			y[j] = s;
		}
		return y;
	}

	/**
	 * Element-wise scalar multiply (allocates a new matrix).
	 *
	 * @param s scalar factor
	 * @return {@code s * this}
	 */
	public CSCMatrix multiply(double s) {
		double[] newData = new double[data.length];
		for (int k = 0; k < data.length; ++k) {
			newData[k] = data[k] * s;
		}
		return new CSCMatrix(rows, cols, indptr.clone(), indices.clone(), newData);
	}

	/**
	 * Transpose. Same nnz, rows and columns swapped. The natural output is
	 * CSR because a CSC column traversal of {@code A} is a CSR row
	 * traversal of {@code A^T}.
	 *
	 * @return {@code A^T} as a {@link CSRMatrix}
	 */
	public CSRMatrix transpose() {
		return new CSRMatrix(cols, rows, indptr.clone(), indices.clone(), data.clone());
	}

	/**
	 * Convert to CSR of the <em>same</em> matrix (not the transpose).
	 *
	 * @return CSR representation of this matrix
	 */
	public CSRMatrix toCSR() {
		int[] outIndptr = new int[rows + 1];
		for (int k = 0; k < indices.length; ++k) {
			outIndptr[indices[k] + 1]++;
		}
		for (int i = 0; i < rows; ++i) {
			outIndptr[i + 1] += outIndptr[i];
		}
		int[] pos = Arrays.copyOf(outIndptr, rows + 1);
		int[] outIndices = new int[indices.length];
		double[] outData = new double[data.length];
		for (int j = 0; j < cols; ++j) {
			for (int k = indptr[j]; k < indptr[j + 1]; ++k) {
				int i = indices[k];
				int dest = pos[i];
				outIndices[dest] = j;
				outData[dest] = data[k];
				pos[i] = dest + 1;
			}
		}
		return new CSRMatrix(rows, cols, outIndptr, outIndices, outData);
	}
}
