package de.labathome;

import org.ujmp.core.Matrix;
import org.ujmp.core.SparseMatrix;

/** linear algebra helper class */
public class LinAlg {

	/**
	 * 2-norm of a vector
	 *
	 * @param v [n] vector
	 * @return sqrt{sum_i{v[i] * v[i]}}
	 */
	public static double norm2(double[] v) {
		double n = 0.0;
		for (int i=0; i<v.length; ++i) {
			n += v[i] * v[i];
		}
		return Math.sqrt(n);
	}

	/**
	 * dot product between to vectors
	 *
	 * @param a [n] a vector
	 * @param b [n] another vector
	 * @return dot product of a and b: sum_i{a[i] * b[i]}
	 */
	public static double dot(double[] a, double[] b) {
		double d = 0.0;
		for (int i=0; i<a.length; ++i) {
			d += a[i] * b[i];
		}
		return d;
	}

	/**
	 * Export the first column as a vector.
	 *
	 * @param a [n][1] "row matrix"
	 * @return [n] first column of a
	 */
	public static double[] col(Matrix a) {
		double[] r = new double[(int) a.getRowCount()];
		for (long[] pos: a.allCoordinates()) {
			r[(int) pos[0]] = a.getAsDouble(pos);
		}
		return r;
	}

	/**
	 * Extract the non-zero elements from a given 2D matrix.
	 *
	 * @param A [n][m] possibly dense matrix
	 * @return [n][m] sparse matrix, contains only non-zero elements of A
	 */
	public static Matrix sparse(Matrix A) {
		Matrix sparseA = SparseMatrix.Factory.zeros(A.getRowCount(), A.getColumnCount());
		for (long[] pos: A.allCoordinates()) {
			double aVal = A.getAsDouble(pos);
			if (aVal != 0.0) {
				sparseA.setAsDouble(aVal, pos);
			}
		}
		return sparseA;
	}
}
