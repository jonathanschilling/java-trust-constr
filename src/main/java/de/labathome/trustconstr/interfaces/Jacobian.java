package de.labathome.trustconstr.interfaces;

import de.labathome.trustconstr.matrix.Matrix;

/**
 * Jacobian evaluator paired with {@link Constraint}: returns the partial
 * derivatives of the canonical-form equality and inequality constraint
 * vectors with respect to {@code x}.
 */
public interface Jacobian {

	/**
	 * @param x current iterate ({@code n &times; 1})
	 * @return Jacobian of {@code constr_eq} at {@code x} ({@code nEq &times; n})
	 */
	public Matrix jacEq(Matrix x);

	/**
	 * @param x current iterate ({@code n &times; 1})
	 * @return Jacobian of {@code constr_ineq} at {@code x}
	 *         ({@code nIneq &times; n})
	 */
	public Matrix jacIneq(Matrix x);
}
