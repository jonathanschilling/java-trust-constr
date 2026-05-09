/*
 * Copyright 2026 Jonathan Schilling
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package de.labathome.trustconstr.matrix;

import dev.ludovic.netlib.blas.BLAS;

/**
 * Dense matrix backed by a column-major {@code double[]} buffer of length
 * {@code rows * cols}. Element {@code (i, j)} is stored at
 * {@code data[i + j * rows]}.
 *
 * <p>Column-major is the layout LAPACK and BLAS expect, so {@link #data()}
 * can be passed directly to {@code dgesv} / {@code dgeqrf} / {@code dgesvd} /
 * {@code dpotrf} / {@code dsyr} / {@code dsyr2} with no copy. The hot kernel
 * {@link #mtimes(Matrix)} dispatches to BLAS {@code dgemm} when both operands
 * are {@code DenseMatrix}; everything else falls back to a triple loop.
 *
 * <p>Subclass of {@link Matrix} so existing call sites that accept a generic
 * {@code Matrix} keep working unchanged. As the migration progresses, callers
 * narrow their declared types to {@code DenseMatrix} where the dense
 * specialisation matters (BLAS dgemm fast path or {@code data()} pass-through
 * to LAPACK).
 */
public final class DenseMatrix extends Matrix {

	private static final BLAS BLAS_INSTANCE = BLAS.getInstance();

	private final int rows;
	private final int cols;
	private final double[] data;

	/**
	 * Allocate a fresh {@code rows x cols} zero-filled {@code DenseMatrix}.
	 *
	 * @param rows number of rows (must be non-negative)
	 * @param cols number of columns (must be non-negative)
	 */
	public DenseMatrix(int rows, int cols) {
		if (rows < 0 || cols < 0) {
			throw new IllegalArgumentException("DenseMatrix dimensions must be non-negative; got "
					+ rows + "x" + cols);
		}
		this.rows = rows;
		this.cols = cols;
		this.data = new double[rows * cols];
	}

	private DenseMatrix(int rows, int cols, double[] data) {
		this.rows = rows;
		this.cols = cols;
		this.data = data;
	}

	/** @return number of rows (primitive {@code int} accessor) */
	public int rows() { return rows; }
	/** @return number of columns (primitive {@code int} accessor) */
	public int cols() { return cols; }

	/**
	 * The underlying column-major buffer. Mutable: callers may overwrite
	 * entries directly (used by BLAS rank updates and LAPACK in-place ops).
	 *
	 * @return the live column-major buffer of length {@code rows * cols};
	 *         entry {@code (i, j)} at index {@code i + j * rows}
	 */
	public double[] data() { return data; }

	/**
	 * @param i 0-based row index
	 * @param j 0-based column index
	 * @return entry at {@code (i, j)}
	 */
	public double get(int i, int j) {
		return data[i + j * rows];
	}

	/**
	 * @param i 0-based row index
	 * @param j 0-based column index
	 * @param v value to store at {@code (i, j)}
	 */
	public void set(int i, int j, double v) {
		data[i + j * rows] = v;
	}

	@Override
	public long getRowCount() { return rows; }
	@Override
	public long getColumnCount() { return cols; }

	@Override
	public double getAsDouble(long row, long col) {
		return data[(int) row + (int) col * rows];
	}

	@Override
	public void setAsDouble(double v, long row, long col) {
		data[(int) row + (int) col * rows] = v;
	}

	@Override
	public DenseMatrix copy() {
		return new DenseMatrix(rows, cols, data.clone());
	}

	@Override
	public double[][] toDoubleArray() {
		double[][] out = new double[rows][cols];
		for (int j = 0; j < cols; ++j) {
			int colOff = j * rows;
			for (int i = 0; i < rows; ++i) {
				out[i][j] = data[i + colOff];
			}
		}
		return out;
	}

	@Override
	public double[] toColumnArray() {
		if (cols != 1) {
			throw new IllegalStateException("toColumnArray() requires a column vector; got "
					+ rows + "x" + cols);
		}
		return data.clone();
	}

	/**
	 * @return contents repacked as a fresh row-major flat {@code double[rows*cols]}
	 *         (entry {@code (i, j)} at index {@code i*cols + j})
	 */
	public double[] toRowMajor() {
		double[] out = new double[rows * cols];
		for (int j = 0; j < cols; ++j) {
			int colOff = j * rows;
			for (int i = 0; i < rows; ++i) {
				out[i * cols + j] = data[i + colOff];
			}
		}
		return out;
	}

	@Override
	public DenseMatrix mtimes(Matrix other) {
		if (other.getRowCount() != cols) {
			throw new IllegalArgumentException("mtimes: incompatible shapes "
					+ rows + "x" + cols + " * "
					+ other.getRowCount() + "x" + other.getColumnCount());
		}
		int p = (int) other.getColumnCount();

		if (other instanceof DenseMatrix b && rows > 0 && cols > 0 && p > 0) {
			DenseMatrix out = new DenseMatrix(rows, p);
			// dgemm: C = alpha A B + beta C, all column-major.
			// A is rowsxcols (lda=rows), B is colsxp (ldb=cols), C is rowsxp (ldc=rows).
			BLAS_INSTANCE.dgemm("N", "N",
					rows, p, cols,
					1.0, this.data, rows,
					b.data, cols,
					0.0, out.data, rows);
			return out;
		}

		// Fallback: dense x non-dense via the abstract getAsDouble.
		DenseMatrix out = new DenseMatrix(rows, p);
		for (int k = 0; k < cols; ++k) {
			int colOffA = k * rows;
			for (int j = 0; j < p; ++j) {
				double bkj = other.getAsDouble(k, j);
				if (bkj == 0.0) continue;
				int colOffC = j * rows;
				for (int i = 0; i < rows; ++i) {
					out.data[i + colOffC] += data[i + colOffA] * bkj;
				}
			}
		}
		return out;
	}

	@Override
	public DenseMatrix plus(Matrix other) {
		checkSameShape(other);
		DenseMatrix out = new DenseMatrix(rows, cols);
		if (other instanceof DenseMatrix b) {
			for (int k = 0; k < data.length; ++k) {
				out.data[k] = data[k] + b.data[k];
			}
		} else {
			for (int j = 0; j < cols; ++j) {
				int colOff = j * rows;
				for (int i = 0; i < rows; ++i) {
					out.data[i + colOff] = data[i + colOff] + other.getAsDouble(i, j);
				}
			}
		}
		return out;
	}

	@Override
	public DenseMatrix minus(Matrix other) {
		checkSameShape(other);
		DenseMatrix out = new DenseMatrix(rows, cols);
		if (other instanceof DenseMatrix b) {
			for (int k = 0; k < data.length; ++k) {
				out.data[k] = data[k] - b.data[k];
			}
		} else {
			for (int j = 0; j < cols; ++j) {
				int colOff = j * rows;
				for (int i = 0; i < rows; ++i) {
					out.data[i + colOff] = data[i + colOff] - other.getAsDouble(i, j);
				}
			}
		}
		return out;
	}

	@Override
	public DenseMatrix times(double alpha) {
		DenseMatrix out = new DenseMatrix(rows, cols);
		for (int k = 0; k < data.length; ++k) {
			out.data[k] = data[k] * alpha;
		}
		return out;
	}

	@Override
	public DenseMatrix times(Matrix elementwise) {
		checkSameShape(elementwise);
		DenseMatrix out = new DenseMatrix(rows, cols);
		if (elementwise instanceof DenseMatrix b) {
			for (int k = 0; k < data.length; ++k) {
				out.data[k] = data[k] * b.data[k];
			}
		} else {
			for (int j = 0; j < cols; ++j) {
				int colOff = j * rows;
				for (int i = 0; i < rows; ++i) {
					out.data[i + colOff] = data[i + colOff] * elementwise.getAsDouble(i, j);
				}
			}
		}
		return out;
	}

	@Override
	public DenseMatrix divide(double alpha) {
		return times(1.0 / alpha);
	}

	@Override
	public DenseMatrix divide(Matrix elementwise) {
		checkSameShape(elementwise);
		DenseMatrix out = new DenseMatrix(rows, cols);
		if (elementwise instanceof DenseMatrix b) {
			for (int k = 0; k < data.length; ++k) {
				out.data[k] = data[k] / b.data[k];
			}
		} else {
			for (int j = 0; j < cols; ++j) {
				int colOff = j * rows;
				for (int i = 0; i < rows; ++i) {
					out.data[i + colOff] = data[i + colOff] / elementwise.getAsDouble(i, j);
				}
			}
		}
		return out;
	}

	@Override
	public DenseMatrix transpose() {
		DenseMatrix out = new DenseMatrix(cols, rows);
		for (int j = 0; j < cols; ++j) {
			int srcOff = j * rows;
			for (int i = 0; i < rows; ++i) {
				out.data[j + i * cols] = data[i + srcOff];
			}
		}
		return out;
	}

	@Override
	public DenseMatrix abs() {
		DenseMatrix out = new DenseMatrix(rows, cols);
		for (int k = 0; k < data.length; ++k) {
			out.data[k] = Math.abs(data[k]);
		}
		return out;
	}

	/**
	 * In-place absolute value.
	 *
	 * @return {@code this} after replacing every entry by its absolute value
	 */
	public DenseMatrix absInPlace() {
		for (int k = 0; k < data.length; ++k) {
			if (data[k] < 0.0) data[k] = -data[k];
		}
		return this;
	}

	/**
	 * In-place scaling.
	 *
	 * @param alpha scalar multiplier
	 * @return {@code this} after multiplying every entry by {@code alpha}
	 */
	public DenseMatrix scaleInPlace(double alpha) {
		for (int k = 0; k < data.length; ++k) {
			data[k] *= alpha;
		}
		return this;
	}

	@Override
	public double norm2() {
		double sum = 0.0;
		for (double v : data) sum += v * v;
		return Math.sqrt(sum);
	}

	@Override
	public double normInf() {
		double max = 0.0;
		for (double v : data) {
			double a = Math.abs(v);
			if (a > max) max = a;
		}
		return max;
	}

	/**
	 * Inclusive submatrix slice {@code rows[r0..r1] x cols[c0..c1]}.
	 * Always returns a fresh {@code DenseMatrix}; no view aliasing.
	 *
	 * @param r0 first row (inclusive)
	 * @param r1 last row (inclusive)
	 * @param c0 first column (inclusive)
	 * @param c1 last column (inclusive)
	 * @return {@code (r1-r0+1) x (c1-c0+1)} slice
	 */
	public DenseMatrix subMatrix(int r0, int r1, int c0, int c1) {
		int nr = r1 - r0 + 1;
		int nc = c1 - c0 + 1;
		DenseMatrix out = new DenseMatrix(nr, nc);
		for (int j = 0; j < nc; ++j) {
			int srcOff = (c0 + j) * rows + r0;
			int dstOff = j * nr;
			System.arraycopy(data, srcOff, out.data, dstOff, nr);
		}
		return out;
	}

	/**
	 * Pick an arbitrary subset of columns into a fresh {@code DenseMatrix}.
	 *
	 * @param colsToKeep column indices to keep, in the desired output order
	 * @return {@code rows x colsToKeep.length} matrix
	 */
	public DenseMatrix selectColumns(int... colsToKeep) {
		DenseMatrix out = new DenseMatrix(rows, colsToKeep.length);
		for (int k = 0; k < colsToKeep.length; ++k) {
			int srcOff = colsToKeep[k] * rows;
			int dstOff = k * rows;
			System.arraycopy(data, srcOff, out.data, dstOff, rows);
		}
		return out;
	}

	/**
	 * Pick an arbitrary subset of rows into a fresh {@code DenseMatrix}.
	 *
	 * @param rowsToKeep row indices to keep, in the desired output order
	 * @return {@code rowsToKeep.length x cols} matrix
	 */
	public DenseMatrix selectRows(int... rowsToKeep) {
		DenseMatrix out = new DenseMatrix(rowsToKeep.length, cols);
		for (int j = 0; j < cols; ++j) {
			int srcOff = j * rows;
			int dstOff = j * rowsToKeep.length;
			for (int i = 0; i < rowsToKeep.length; ++i) {
				out.data[dstOff + i] = data[srcOff + rowsToKeep[i]];
			}
		}
		return out;
	}

	private void checkSameShape(Matrix other) {
		if (other.getRowCount() != rows || other.getColumnCount() != cols) {
			throw new IllegalArgumentException("shape mismatch: "
					+ rows + "x" + cols + " vs "
					+ other.getRowCount() + "x" + other.getColumnCount());
		}
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("DenseMatrix ").append(rows).append('x').append(cols).append(" [");
		for (int i = 0; i < rows; ++i) {
			if (i > 0) sb.append("; ");
			for (int j = 0; j < cols; ++j) {
				if (j > 0) sb.append(", ");
				sb.append(data[i + j * rows]);
			}
		}
		sb.append(']');
		return sb.toString();
	}

	// --- Static factories ---

	/**
	 * @param rows number of rows (must be non-negative)
	 * @param cols number of columns (must be non-negative)
	 * @return a fresh {@code rows x cols} matrix filled with zeros
	 */
	public static DenseMatrix zeros(int rows, int cols) {
		return new DenseMatrix(rows, cols);
	}

	/**
	 * @param n dimension (must be non-negative)
	 * @return the {@code n x n} identity matrix
	 */
	public static DenseMatrix eye(int n) {
		DenseMatrix m = new DenseMatrix(n, n);
		for (int i = 0; i < n; ++i) {
			m.data[i + i * n] = 1.0;
		}
		return m;
	}

	/**
	 * @param v entries of the resulting column vector (copied)
	 * @return an {@code n x 1} column vector
	 */
	public static DenseMatrix column(double... v) {
		double[] copy = v.clone();
		return new DenseMatrix(v.length, 1, copy);
	}

	/**
	 * @param v entries of the resulting row vector (copied)
	 * @return a {@code 1 x n} row vector
	 */
	public static DenseMatrix row(double... v) {
		double[] copy = v.clone();
		return new DenseMatrix(1, v.length, copy);
	}

	/**
	 * Repack a row-major {@code double[][]} into a fresh column-major
	 * {@code DenseMatrix}.
	 *
	 * @param rows row-major source ({@code rows[i][j]} = entry at {@code (i,j)})
	 * @return a fresh {@code DenseMatrix} with the same shape and values
	 */
	public static DenseMatrix fromRows(double[][] rows) {
		int n = rows.length;
		int m = n == 0 ? 0 : rows[0].length;
		DenseMatrix out = new DenseMatrix(n, m);
		for (int i = 0; i < n; ++i) {
			if (rows[i].length != m) {
				throw new IllegalArgumentException("fromRows: row " + i + " has length "
						+ rows[i].length + ", expected " + m);
			}
			for (int j = 0; j < m; ++j) {
				out.data[i + j * n] = rows[i][j];
			}
		}
		return out;
	}

	/**
	 * Wrap an already-column-major buffer without copying. The caller surrenders
	 * ownership of {@code data}: subsequent mutations through this matrix will
	 * be visible to anyone else holding the array.
	 *
	 * @param rows number of rows
	 * @param cols number of columns
	 * @param data column-major buffer of length {@code rows * cols}; the
	 *             entry at {@code (i, j)} is {@code data[i + j * rows]}
	 * @return a {@code DenseMatrix} that aliases {@code data}
	 */
	public static DenseMatrix fromColumnMajor(int rows, int cols, double[] data) {
		if (data.length != rows * cols) {
			throw new IllegalArgumentException("fromColumnMajor: data length " + data.length
					+ " != rows*cols (" + rows + "*" + cols + " = " + (rows * cols) + ")");
		}
		return new DenseMatrix(rows, cols, data);
	}

	/**
	 * Densify any {@link Matrix}: if {@code m} is already a {@code DenseMatrix},
	 * return a {@link #copy()}; otherwise materialise via
	 * {@link #getAsDouble(long, long)}.
	 *
	 * @param m source matrix
	 * @return a fresh dense copy
	 */
	public static DenseMatrix copyFromMatrix(Matrix m) {
		if (m instanceof DenseMatrix d) {
			return d.copy();
		}
		int rows = (int) m.getRowCount();
		int cols = (int) m.getColumnCount();
		DenseMatrix out = new DenseMatrix(rows, cols);
		for (int j = 0; j < cols; ++j) {
			int colOff = j * rows;
			for (int i = 0; i < rows; ++i) {
				out.data[i + colOff] = m.getAsDouble(i, j);
			}
		}
		return out;
	}

	/**
	 * Backwards-compatible nested factory class. New code should call the
	 * top-level static factories on {@link DenseMatrix} directly; this exists
	 * only so existing callers like {@code DenseMatrix.Factory.copyFromMatrix(m)}
	 * keep compiling during the staged migration.
	 */
	public static final class Factory {
		private Factory() {}

		/**
		 * @param m source matrix
		 * @return a fresh dense copy; alias for {@link DenseMatrix#copyFromMatrix(Matrix)}
		 */
		public static DenseMatrix copyFromMatrix(Matrix m) {
			return DenseMatrix.copyFromMatrix(m);
		}
	}
}
