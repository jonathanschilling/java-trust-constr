package org.scipy.optimize.minimize.interfaces;

import org.scipy.optimize.minimize.matrix.Matrix;

/**
 * Evaluator for the canonical-form constraints of the optimization problem:
 * {@code constr_eq(x) = 0} and {@code constr_ineq(x) &le; 0}.
 */
public interface Constraint {

	/**
	 * @param x current iterate ({@code n &times; 1})
	 * @return equality-constraint residual ({@code nEq &times; 1})
	 */
	public Matrix constrEq(Matrix x);

	/**
	 * @param x current iterate ({@code n &times; 1})
	 * @return inequality-constraint residual ({@code nIneq &times; 1});
	 *         feasible iff every entry is {@code &le; 0}
	 */
	public Matrix constrIneq(Matrix x);
}
