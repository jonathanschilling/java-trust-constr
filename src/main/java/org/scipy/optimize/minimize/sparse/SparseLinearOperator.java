package org.scipy.optimize.minimize.sparse;

import org.scipy.optimize.minimize.LinAlg;
import org.scipy.optimize.minimize.interfaces.LinearOperator;
import org.ujmp.core.Matrix;

/**
 * Adapter from a {@link CSRMatrix} or {@link CSCMatrix} to the existing
 * {@link LinearOperator} interface used throughout the algorithm.
 *
 * <p>Mirrors the role of {@code scipy.sparse.linalg.LinearOperator} produced
 * from a sparse matrix via {@code aslinearoperator(A)}.
 *
 * <p>The {@link LinearOperator} interface is currently typed in terms of UJMP
 * {@link Matrix}, so this class does the array<->matrix conversion at the
 * boundary. When the project finishes its UJMP -> ojAlgo migration, this
 * adapter is the single point that needs to follow.
 */
public final class SparseLinearOperator {

	private SparseLinearOperator() { }

	/** {@code y = A x}. */
	public static LinearOperator forMatvec(CSRMatrix a) {
		return x -> {
			double[] xArr = LinAlg.col(x);
			double[] yArr = a.matvec(xArr);
			return Matrix.Factory.linkToArray(yArr).transpose();
		};
	}

	/** {@code y = A x}. */
	public static LinearOperator forMatvec(CSCMatrix a) {
		return x -> {
			double[] xArr = LinAlg.col(x);
			double[] yArr = a.matvec(xArr);
			return Matrix.Factory.linkToArray(yArr).transpose();
		};
	}

	/** {@code y = A^T x}. */
	public static LinearOperator forRmatvec(CSRMatrix a) {
		return x -> {
			double[] xArr = LinAlg.col(x);
			double[] yArr = a.rmatvec(xArr);
			return Matrix.Factory.linkToArray(yArr).transpose();
		};
	}

	/** {@code y = A^T x}. */
	public static LinearOperator forRmatvec(CSCMatrix a) {
		return x -> {
			double[] xArr = LinAlg.col(x);
			double[] yArr = a.rmatvec(xArr);
			return Matrix.Factory.linkToArray(yArr).transpose();
		};
	}
}
