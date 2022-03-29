package org.scipy.optimize.minimize.interfaces;

import org.scipy.optimize.minimize.FunctionAndConstraint;
import org.ujmp.core.Matrix;

@FunctionalInterface
public interface IFunctionAndConstraint {
	public FunctionAndConstraint funAndConstr(Matrix x);
}
