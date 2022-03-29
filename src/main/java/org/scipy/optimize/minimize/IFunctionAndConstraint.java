package org.scipy.optimize.minimize;

import org.ujmp.core.Matrix;

@FunctionalInterface
public interface IFunctionAndConstraint {
	public FunctionAndConstraint funAndConstr(Matrix x);
}
