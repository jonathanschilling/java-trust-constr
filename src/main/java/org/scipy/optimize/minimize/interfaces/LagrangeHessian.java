package org.scipy.optimize.minimize.interfaces;

import org.ujmp.core.Matrix;

@FunctionalInterface
public interface LagrangeHessian {

	/**
	 * Compute Hessian of constraint.
	 *
	 * @param x [n] position
	 * @param v [m] Lagrange multipliers
	 * @return [n][n] Hessian of constraint
	 */
	public LinearOperator lagrHess(Matrix x, Matrix v);

}
