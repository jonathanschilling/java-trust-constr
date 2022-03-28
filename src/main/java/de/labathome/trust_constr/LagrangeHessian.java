package de.labathome.trust_constr;

import org.ujmp.core.Matrix;

public interface LagrangeHessian {

	/**
	 * Compute Hessian of constraint.
	 *
	 * @param x [n] position
	 * @param v [m] Lagrange multipliers
	 * @return [n][n] Hessian of constraint
	 */
	public Matrix lagrHess(Matrix x, Matrix v);

}
