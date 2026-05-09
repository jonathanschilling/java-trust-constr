package org.scipy.optimize.minimize.sparse;

import org.scipy.optimize.minimize.matrix.MatrixOps;
import org.scipy.optimize.minimize.interfaces.LinearOperator;
import org.scipy.optimize.minimize.matrix.Matrix;

/**
 * Adapter from a {@link CSRMatrix} or {@link CSCMatrix} to the existing
 * {@link LinearOperator} interface used throughout the algorithm. Mirrors
 * the role of {@code scipy.sparse.linalg.LinearOperator} produced from a
 * sparse matrix via {@code aslinearoperator(A)}.
 *
 * <p>The {@link LinearOperator} contract is typed in terms of {@link Matrix},
 * so this adapter does the {@code double[]} &harr; {@link Matrix} conversion
 * at the boundary on each apply.
 */
public final class SparseLinearOperator {

	private SparseLinearOperator() { }

	/**
	 * @param a CSR matrix
	 * @return a {@link LinearOperator} that computes {@code y = A x}
	 */
	public static LinearOperator forMatvec(CSRMatrix a) {
		return x -> {
			double[] xArr = x.toColumnArray();
			double[] yArr = a.matvec(xArr);
			return Matrix.Factory.linkToArray(yArr).transpose();
		};
	}

	/**
	 * @param a CSC matrix
	 * @return a {@link LinearOperator} that computes {@code y = A x}
	 */
	public static LinearOperator forMatvec(CSCMatrix a) {
		return x -> {
			double[] xArr = x.toColumnArray();
			double[] yArr = a.matvec(xArr);
			return Matrix.Factory.linkToArray(yArr).transpose();
		};
	}

	/**
	 * @param a CSR matrix
	 * @return a {@link LinearOperator} that computes {@code y = A^T x}
	 */
	public static LinearOperator forRmatvec(CSRMatrix a) {
		return x -> {
			double[] xArr = x.toColumnArray();
			double[] yArr = a.rmatvec(xArr);
			return Matrix.Factory.linkToArray(yArr).transpose();
		};
	}

	/**
	 * @param a CSC matrix
	 * @return a {@link LinearOperator} that computes {@code y = A^T x}
	 */
	public static LinearOperator forRmatvec(CSCMatrix a) {
		return x -> {
			double[] xArr = x.toColumnArray();
			double[] yArr = a.rmatvec(xArr);
			return Matrix.Factory.linkToArray(yArr).transpose();
		};
	}
}
