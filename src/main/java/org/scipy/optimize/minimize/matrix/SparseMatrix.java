package org.scipy.optimize.minimize.matrix;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import org.scipy.optimize.minimize.sparse.CSRMatrix;

/**
 * Sparse matrix backed by a Dictionary-of-Keys (DOK) representation:
 * a {@link HashMap} keyed by the encoded {@code (row, col)} index. Supports
 * cheap incremental {@link #setAsDouble} (the equivalent of scipy's
 * {@code lil_matrix} build phase) and converts to
 * {@link CSRMatrix} for compute-heavy
 * paths via {@link #toCSR()}.
 *
 * <p>Compute kernels ({@link #mtimes(Matrix)}, {@link #plus(Matrix)} etc.)
 * materialise to dense or convert to CSR internally -- no attempt is made to
 * keep results sparse. The trust-constr port only relies on sparsity in a
 * handful of well-known places (Jacobian storage and the AUGMENTED_SYSTEM
 * KKT factorization, both of which already drop into the CSR fast paths).
 */
public class SparseMatrix extends Matrix {

	private final int rows;
	private final int cols;
	private final HashMap<Long, Double> entries;

	/**
	 * Allocate a fresh empty sparse matrix.
	 *
	 * @param rows number of rows
	 * @param cols number of columns
	 */
	public SparseMatrix(int rows, int cols) {
		this.rows = rows;
		this.cols = cols;
		this.entries = new HashMap<>();
	}

	@Override
	public boolean isSparse() {
		return true;
	}

	@Override
	public long getRowCount() { return rows; }
	@Override
	public long getColumnCount() { return cols; }

	@Override
	public double getAsDouble(long row, long col) {
		Double v = entries.get(key((int) row, (int) col));
		return v == null ? 0.0 : v;
	}

	@Override
	public void setAsDouble(double v, long row, long col) {
		long k = key((int) row, (int) col);
		if (v == 0.0) {
			entries.remove(k);
		} else {
			entries.put(k, v);
		}
	}

	private long key(int row, int col) {
		return ((long) row) * cols + col;
	}

	/** @return number of stored (non-zero) entries */
	public int nnz() {
		return entries.size();
	}

	/**
	 * Materialise this sparse matrix to a
	 * {@link CSRMatrix} (canonical CSR
	 * form: column indices sorted within each row, no explicit zeros).
	 *
	 * @return a CSR view of the same content
	 */
	public CSRMatrix toCSR() {
		// Two-pass: count nnz per row to size indptr/indices/data.
		int[] rowCount = new int[rows];
		for (Long k : entries.keySet()) {
			int r = (int) (k / cols);
			++rowCount[r];
		}
		int[] indptr = new int[rows + 1];
		for (int i = 0; i < rows; ++i) {
			indptr[i + 1] = indptr[i] + rowCount[i];
		}
		int total = indptr[rows];
		int[] indices = new int[total];
		double[] data = new double[total];
		int[] cursor = indptr.clone();
		// Sort by (row, col) within each row by inserting in column-ascending order.
		// First pass: collect keys per row.
		ArrayList<long[]> perRow = new ArrayList<>(rows);
		for (int i = 0; i < rows; ++i) {
			perRow.add(new long[rowCount[i] * 2]);  // pairs (col, encoded key)
		}
		int[] perRowFill = new int[rows];
		for (Map.Entry<Long, Double> e : entries.entrySet()) {
			long k = e.getKey();
			int r = (int) (k / cols);
			int c = (int) (k % cols);
			long[] arr = perRow.get(r);
			int idx = perRowFill[r]++;
			arr[idx * 2] = c;
			arr[idx * 2 + 1] = Double.doubleToRawLongBits(e.getValue());
		}
		for (int i = 0; i < rows; ++i) {
			long[] arr = perRow.get(i);
			int n = rowCount[i];
			// Insertion sort by column.
			for (int p = 1; p < n; ++p) {
				long c = arr[p * 2];
				long v = arr[p * 2 + 1];
				int q = p - 1;
				while (q >= 0 && arr[q * 2] > c) {
					arr[(q + 1) * 2] = arr[q * 2];
					arr[(q + 1) * 2 + 1] = arr[q * 2 + 1];
					--q;
				}
				arr[(q + 1) * 2] = c;
				arr[(q + 1) * 2 + 1] = v;
			}
			for (int p = 0; p < n; ++p) {
				int slot = cursor[i]++;
				indices[slot] = (int) arr[p * 2];
				data[slot] = Double.longBitsToDouble(arr[p * 2 + 1]);
			}
		}
		return new CSRMatrix(rows, cols, indptr, indices, data);
	}

	@Override
	public Matrix copy() {
		SparseMatrix out = new SparseMatrix(rows, cols);
		out.entries.putAll(entries);
		return out;
	}

	@Override
	public Matrix mtimes(Matrix other) {
		// Materialise via CSR matvec when other is a column vector (common case).
		if (other.getColumnCount() == 1) {
			double[] x = new double[(int) other.getRowCount()];
			for (int i = 0; i < x.length; ++i) {
				x[i] = other.getAsDouble(i, 0);
			}
			double[] y = toCSR().matvec(x);
			DenseMatrix out = DenseMatrix.zeros(rows, 1);
			for (int i = 0; i < rows; ++i) {
				out.set(i, 0, y[i]);
			}
			return out;
		}
		// Fallback: dense triple loop.
		int p = (int) other.getColumnCount();
		DenseMatrix out = DenseMatrix.zeros(rows, p);
		for (Map.Entry<Long, Double> e : entries.entrySet()) {
			long k = e.getKey();
			int r = (int) (k / cols);
			int c = (int) (k % cols);
			double v = e.getValue();
			for (int j = 0; j < p; ++j) {
				out.set(r, j, out.get(r, j) + v * other.getAsDouble(c, j));
			}
		}
		return out;
	}

	@Override
	public Matrix plus(Matrix other) {
		return densify().plus(other);
	}

	@Override
	public Matrix minus(Matrix other) {
		return densify().minus(other);
	}

	@Override
	public Matrix times(double alpha) {
		SparseMatrix out = new SparseMatrix(rows, cols);
		for (Map.Entry<Long, Double> e : entries.entrySet()) {
			out.entries.put(e.getKey(), e.getValue() * alpha);
		}
		return out;
	}

	@Override
	public Matrix times(Matrix elementwise) {
		return densify().times(elementwise);
	}

	@Override
	public Matrix divide(double alpha) {
		return times(1.0 / alpha);
	}

	@Override
	public Matrix divide(Matrix elementwise) {
		return densify().divide(elementwise);
	}

	@Override
	public Matrix transpose() {
		SparseMatrix out = new SparseMatrix(cols, rows);
		for (Map.Entry<Long, Double> e : entries.entrySet()) {
			long k = e.getKey();
			int r = (int) (k / cols);
			int c = (int) (k % cols);
			out.entries.put(((long) c) * rows + r, e.getValue());
		}
		return out;
	}

	@Override
	public Matrix abs() {
		SparseMatrix out = new SparseMatrix(rows, cols);
		for (Map.Entry<Long, Double> e : entries.entrySet()) {
			out.entries.put(e.getKey(), Math.abs(e.getValue()));
		}
		return out;
	}

	@Override
	public double norm2() {
		double sum = 0.0;
		for (Double v : entries.values()) {
			sum += v * v;
		}
		return Math.sqrt(sum);
	}

	@Override
	public double normInf() {
		double max = 0.0;
		for (Double v : entries.values()) {
			double a = Math.abs(v);
			if (a > max) max = a;
		}
		return max;
	}

	private DenseMatrix densify() {
		DenseMatrix d = DenseMatrix.zeros(rows, cols);
		for (Map.Entry<Long, Double> e : entries.entrySet()) {
			long k = e.getKey();
			d.set((int) (k / cols), (int) (k % cols), e.getValue());
		}
		return d;
	}

	/** Static factories that produce {@link SparseMatrix} instances. */
	public static final class Factory {

		private Factory() {}

		/**
		 * @param rows number of rows
		 * @param cols number of columns
		 * @return a fresh empty {@link SparseMatrix}
		 */
		public static SparseMatrix zeros(long rows, long cols) {
			return new SparseMatrix((int) rows, (int) cols);
		}

		/**
		 * @param size length-2 shape array {@code [rows, cols]}
		 * @return a fresh empty {@link SparseMatrix}
		 */
		public static SparseMatrix zeros(long[] size) {
			long r = size.length > 0 ? size[0] : 0;
			long c = size.length > 1 ? size[1] : 1;
			return zeros(r, c);
		}

		/**
		 * @param m source matrix
		 * @return a fresh {@link SparseMatrix} containing the non-zero
		 *         entries of {@code m}
		 */
		public static SparseMatrix copyFromMatrix(Matrix m) {
			SparseMatrix out = new SparseMatrix((int) m.getRowCount(), (int) m.getColumnCount());
			int rows = (int) m.getRowCount();
			int cols = (int) m.getColumnCount();
			for (int i = 0; i < rows; ++i) {
				for (int j = 0; j < cols; ++j) {
					double v = m.getAsDouble(i, j);
					if (v != 0.0) {
						out.setAsDouble(v, i, j);
					}
				}
			}
			return out;
		}
	}
}
