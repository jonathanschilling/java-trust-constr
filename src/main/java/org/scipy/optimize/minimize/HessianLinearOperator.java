package org.scipy.optimize.minimize;

import java.util.function.BiFunction;

import org.scipy.optimize.minimize.interfaces.HessianProduct;
import org.scipy.optimize.minimize.matrix.Matrix;

/**
 * Adapter that exposes a matrix-free {@link HessianProduct} as a
 * {@link BiFunction} returning an explicit Hessian {@link Matrix}.
 *
 * <p>Mirrors scipy's {@code LinearOperator}-wrapping of {@code hessp} in
 * {@code _minimize_trustregion_constr} ({@code minimize_trustregion_constr.py:36}):
 * given a callable {@code hessp(x, p)} returning {@code H(x)·p} for each
 * vector {@code p}, the wrapper materialises {@code H(x)} as a dense
 * {@code n x n} matrix by probing {@code hessp} along each Cartesian basis
 * vector. This is O(n) hessp evaluations — sufficient for the modest
 * problem sizes trust-constr is typically used on, and a future hook for
 * a true matrix-free path when downstream consumers can accept one.
 */
public class HessianLinearOperator implements BiFunction<Matrix, Object, Matrix> {

	private final HessianProduct hessp;
	private final long nVars;

	public HessianLinearOperator(HessianProduct hessp, long nVars) {
		this.hessp = hessp;
		this.nVars = nVars;
	}

	@Override
	public Matrix apply(Matrix x, Object args) {
		int n = (int) nVars;
		double[][] hDense = new double[n][n];
		Matrix p = Matrix.Factory.zeros(n, 1);
		for (int j = 0; j < n; ++j) {
			if (j > 0) {
				p.setAsDouble(0.0, j - 1, 0);
			}
			p.setAsDouble(1.0, j, 0);
			Matrix hp = hessp.apply(x, p, args);
			for (int i = 0; i < n; ++i) {
				hDense[i][j] = hp.getAsDouble(i, 0);
			}
		}
		return Matrix.Factory.linkToArray(hDense);
	}
}
