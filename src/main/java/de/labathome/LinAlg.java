package de.labathome;

/** linear algebra helper class */
public class LinAlg {

	/**
	 * Compute the Frobenius norm for a given matrix.
	 *
	 * @param A [n][m] matrix
	 * @return Frobenius norm of A: sqrt{sum_ij{A_ij^2}}
	 */
	public static double frob(double[][] A) {
		double f = 0.0;
		for (int i=0; i<A.length; ++i) {
			for (int j=0; j<A[0].length; ++j) {
				f += A[i][j] * A[i][j];
			}
		}
		return Math.sqrt(f);
	}

	/**
	 * element-wise sum of two vectors
	 * @param a [n] a vector
	 * @param b [n] another vector
	 * @return [n] the i-th element is a[i] + b[i] for i=0,1,...,(n-1)
	 */
	public static double[] add(double[] a, double[] b) {
		int n = a.length;
		double[] result = new double[n];
		for (int i=0; i<n; ++i) {
			result[i] = a[i] + b[i];
		}
		return result;
	}

	/**
	 * element-wise difference between two vectors
	 * @param a [n] a vector
	 * @param b [n] another vector
	 * @return [n] the i-th element is a[i] - b[i] for i=0,1,...,(n-1)
	 */
	public static double[] subtract(double[] a, double[] b) {
		int n = a.length;
		double[] result = new double[n];
		for (int i=0; i<n; ++i) {
			result[i] = a[i] - b[i];
		}
		return result;
	}

	/**
	 * element-wise multiply
	 *
	 * @param a [n] vector
	 * @param scale scaling factor
	 * @return [n] a*scale element-wise
	 */
	public static double[] mulElem(double[] a, double scale) {
		int n = a.length;
		double[] scaledA = new double[n];
		for (int i=0; i<n; ++i) {
			scaledA[i] = a[i] * scale;
		}
		return scaledA;
	}

	/**
	 * matrix-vector product
	 *
	 * @param A [n][m] matrix
	 * @param b [m] vector
	 * @return [n] result vector = A * b
	 */
	public static double[] dot(double[][] A, double[] b) {
		double scale = 1.0;
		return dot(A, b, scale);
	}

	/**
	 * matrix-vector product
	 *
	 * @param A [n][m] matrix
	 * @param b [m] vector
	 * @param scale scaling factor to apply along the way
	 * @return [n] result vector = A * b * scale
	 */
	public static double[] dot(double[][] A, double[] b, double scale) {
		boolean transposeA = false;
		return dot(A, transposeA, b, scale);
	}

	/**
	 * matrix-vector product
	 *
	 * @param A [n][m] matrix
	 * @param transposeA if true, assume A has shape [m][n]; otherwise, assume A has shape [n][m]
	 * @param b [m] vector
	 * @return [n] result vector = A * b or A^T * b
	 */
	public static double[] dot(double[][] A, boolean transposeA, double[] b) {
		double scale = 1.0;
		return dot(A, transposeA, b, scale);
	}

	/**
	 * matrix-vector product
	 *
	 * @param A [n][m] matrix
	 * @param transposeA if true, assume A has shape [m][n]; otherwise, assume A has shape [n][m]
	 * @param b [m] vector
	 * @param scale scaling factor to apply along the way
	 * @return [n] result vector = A * b * scale or A^T * b * scale
	 */
	public static double[] dot(double[][] A, boolean transposeA, double[] b, double scale) {
		int n = A.length;
		int m = A[0].length;
		double[] result = new double[n];
		if (!transposeA) {
			for (int i=0; i<n; ++i) {
				double r = 0.0;
				double[] aRow = A[i];
				for (int j=0; j<m; ++j) {
					r += aRow[j] * b[j];
				}
				result[i] = r * scale;
			}
		} else {
			for (int j=0; j<m; ++j) {
				double scaledB = b[j] * scale;
				double[] aRow = A[j];
				for (int i=0; i<n; ++i) {
					result[i] += aRow[i] * scaledB;
				}
			}
		}
		return result;
	}

	/**
	 * 2-norm of a vector
	 *
	 * @param v [n] vector
	 * @return sqrt{sum_i{v[i] * v[i]}}
	 */
	public static double norm(double[] v) {
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
}
