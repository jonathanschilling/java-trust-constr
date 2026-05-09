package de.labathome.trustconstr.interfaces;

import de.labathome.trustconstr.records.FunctionAndConstraint;
import de.labathome.trustconstr.matrix.Matrix;

/** Combined evaluator returning {@code (f(x), constr(x))} in a single call. */
@FunctionalInterface
public interface IFunctionAndConstraint {

	/**
	 * @param x current iterate ({@code n &times; 1})
	 * @return objective value and constraint vector at {@code x}
	 */
	public FunctionAndConstraint funAndConstr(Matrix x);
}
