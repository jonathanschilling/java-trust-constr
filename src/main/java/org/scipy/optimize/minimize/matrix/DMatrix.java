package org.scipy.optimize.minimize.matrix;

import java.util.Arrays;

import dev.ludovic.netlib.blas.BLAS;

/**
 * Dense matrix backed by row-major {@code double[][]}. Hot kernels
 * ({@link #mtimes(Matrix)}) dispatch to {@code dev.ludovic.netlib} BLAS
 * {@code dgemm}; everything else is plain Java loops.
 */
public class DMatrix extends Matrix {

	private static final BLAS BLAS_INSTANCE = BLAS.getInstance();

	final double[][] data;
	private final int rows;
	private final int cols;

	public DMatrix(int rows, int cols) {
		this.rows = rows;
		this.cols = cols;
		this.data = new double[rows][cols];
	}

	public DMatrix(double[][] data) {
		this.data = data;
		this.rows = data.length;
		this.cols = rows == 0 ? 0 : data[0].length;
	}

	public int rows() { return rows; }
	public int cols() { return cols; }

	public double[] rowData(int i) {
		return data[i];
	}

	public double get(int i, int j) {
		return data[i][j];
	}

	public void set(int i, int j, double v) {
		data[i][j] = v;
	}

	@Override
	public long getRowCount() { return rows; }
	@Override
	public long getColumnCount() { return cols; }

	@Override
	public double getAsDouble(long row, long col) {
		return data[(int) row][(int) col];
	}

	@Override
	public void setAsDouble(double v, long row, long col) {
		data[(int) row][(int) col] = v;
	}

	@Override
	public double[][] toDoubleArray() {
		double[][] out = new double[rows][];
		for (int i = 0; i < rows; ++i) {
			out[i] = data[i].clone();
		}
		return out;
	}

	@Override
	public Matrix copy() {
		DMatrix out = new DMatrix(rows, cols);
		for (int i = 0; i < rows; ++i) {
			System.arraycopy(data[i], 0, out.data[i], 0, cols);
		}
		return out;
	}

	@Override
	public Matrix mtimes(Matrix other) {
		int p = (int) other.getColumnCount();
		if (other.getRowCount() != cols) {
			throw new IllegalArgumentException("mtimes: incompatible shapes "
					+ rows + "x" + cols + " * "
					+ other.getRowCount() + "x" + other.getColumnCount());
		}
		// Dense-dense path: BLAS dgemm.
		if (other instanceof DMatrix && rows > 0 && cols > 0 && p > 0) {
			DMatrix b = (DMatrix) other;
			// Both arrays are row-major; BLAS expects column-major. We compute
			// C^T = B^T A^T in column-major terms by passing A and B as
			// transposed (column-major) views of their row-major data.
			double[] aFlat = flattenRowMajor(this);
			double[] bFlat = flattenRowMajor(b);
			double[] cFlat = new double[rows * p];
			// BLAS sees aFlat as A^T (column-major, shape cols x rows),
			// bFlat as B^T (column-major, shape p x cols).
			// We want C = A * B (row-major, shape rows x p), i.e.
			// C^T = B^T * A^T (column-major, shape p x rows).
			// dgemm("N", "N", p, rows, cols, 1, B^T, p, A^T, cols, 0, C^T, p).
			BLAS_INSTANCE.dgemm("N", "N", p, rows, cols, 1.0, bFlat, p, aFlat, cols, 0.0, cFlat, p);
			// cFlat is C^T column-major (shape p x rows), i.e. row-major C
			// (shape rows x p) flattened.
			DMatrix out = new DMatrix(rows, p);
			for (int i = 0; i < rows; ++i) {
				System.arraycopy(cFlat, i * p, out.data[i], 0, p);
			}
			return out;
		}
		// Fallback: dense × any. Simple triple-loop using getAsDouble.
		DMatrix out = new DMatrix(rows, p);
		for (int i = 0; i < rows; ++i) {
			for (int k = 0; k < cols; ++k) {
				double aik = data[i][k];
				if (aik == 0.0) continue;
				for (int j = 0; j < p; ++j) {
					out.data[i][j] += aik * other.getAsDouble(k, j);
				}
			}
		}
		return out;
	}

	@Override
	public Matrix plus(Matrix other) {
		checkSameShape(other);
		int n = rows, m = cols;
		DMatrix out = new DMatrix(n, m);
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				out.data[i][j] = data[i][j] + other.getAsDouble(i, j);
			}
		}
		return out;
	}

	@Override
	public Matrix minus(Matrix other) {
		checkSameShape(other);
		int n = rows, m = cols;
		DMatrix out = new DMatrix(n, m);
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				out.data[i][j] = data[i][j] - other.getAsDouble(i, j);
			}
		}
		return out;
	}

	@Override
	public Matrix times(double alpha) {
		DMatrix out = new DMatrix(rows, cols);
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				out.data[i][j] = data[i][j] * alpha;
			}
		}
		return out;
	}

	@Override
	public Matrix times(Matrix elementwise) {
		checkSameShape(elementwise);
		DMatrix out = new DMatrix(rows, cols);
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				out.data[i][j] = data[i][j] * elementwise.getAsDouble(i, j);
			}
		}
		return out;
	}

	@Override
	public Matrix divide(double alpha) {
		return times(1.0 / alpha);
	}

	@Override
	public Matrix divide(Matrix elementwise) {
		checkSameShape(elementwise);
		DMatrix out = new DMatrix(rows, cols);
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				out.data[i][j] = data[i][j] / elementwise.getAsDouble(i, j);
			}
		}
		return out;
	}

	@Override
	public Matrix transpose() {
		DMatrix out = new DMatrix(cols, rows);
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				out.data[j][i] = data[i][j];
			}
		}
		return out;
	}

	@Override
	public Matrix abs() {
		DMatrix out = new DMatrix(rows, cols);
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				out.data[i][j] = Math.abs(data[i][j]);
			}
		}
		return out;
	}

	@Override
	public double norm2() {
		// Frobenius norm.
		double sum = 0.0;
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				double v = data[i][j];
				sum += v * v;
			}
		}
		return Math.sqrt(sum);
	}

	@Override
	public double normInf() {
		double max = 0.0;
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				double v = Math.abs(data[i][j]);
				if (v > max) max = v;
			}
		}
		return max;
	}

	private void checkSameShape(Matrix other) {
		if (other.getRowCount() != rows || other.getColumnCount() != cols) {
			throw new IllegalArgumentException("shape mismatch: "
					+ rows + "x" + cols + " vs "
					+ other.getRowCount() + "x" + other.getColumnCount());
		}
	}

	private static double[] flattenRowMajor(DMatrix m) {
		double[] out = new double[m.rows * m.cols];
		for (int i = 0; i < m.rows; ++i) {
			System.arraycopy(m.data[i], 0, out, i * m.cols, m.cols);
		}
		return out;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("DMatrix ").append(rows).append('x').append(cols).append(' ');
		sb.append(Arrays.deepToString(data));
		return sb.toString();
	}
}
