package org.scipy.optimize.minimize;

import java.util.function.ToDoubleFunction;

import org.scipy.optimize.minimize.interfaces.LinearOperator;
import org.scipy.optimize.minimize.records.FiniteDifferenceOptions;
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
	 * @param A [n][1] "row matrix"
	 * @return [n] first column of a
	 */
	public static double[] col(Matrix A) {
		return A.transpose().toDoubleArray()[0];
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

	public static Matrix diag(double[] diagonal) {
		int n = diagonal.length;
		Matrix d = SparseMatrix.Factory.zeros(n, n);
		for (int i=0; i<n; ++i) {
			d.setAsDouble(diagonal[i], i, i);
		}
		return d;
	}

	public static LinearOperator op(Matrix A) {
		return new LinearOperator() {
			@Override
			public Matrix apply(Matrix x) {
				return A.mtimes(x);
			}
		};
	}

	/**
	 * finite-difference gradient
	 *
	 * @param f
	 * @param x
	 * @param f0
	 * @param options
	 * @return
	 */
	public static Matrix approxDerivative(ToDoubleFunction<Matrix> f, Matrix x, double f0, FiniteDifferenceOptions options) {

		// TODO

		return null;
	}

	/**
	 * finite-difference Hessian
	 *
	 * @param f
	 * @param x
	 * @param f0
	 * @param options
	 * @return
	 */
	public static Matrix approxDerivative(java.util.function.Function<Matrix, Matrix> f, Matrix x, Matrix f0, FiniteDifferenceOptions options) {

		// TODO

		return null;
	}

}
