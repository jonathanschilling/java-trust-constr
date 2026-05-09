package org.scipy.optimize.minimize.interfaces;

import org.scipy.optimize.minimize.records.FunctionAndConstraint;
import org.scipy.optimize.minimize.matrix.Matrix;

/** Combined evaluator returning {@code (f(x), constr(x))} in a single call. */
@FunctionalInterface
public interface IFunctionAndConstraint {

	/**
	 * @param x current iterate ({@code n &times; 1})
	 * @return objective value and constraint vector at {@code x}
	 */
	public FunctionAndConstraint funAndConstr(Matrix x);
}
