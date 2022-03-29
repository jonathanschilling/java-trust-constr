package org.scipy.optimize.minimize.interfaces;

import org.scipy.optimize.minimize.records.FunctionAndConstraint;
import org.ujmp.core.Matrix;

@FunctionalInterface
public interface IFunctionAndConstraint {
	public FunctionAndConstraint funAndConstr(Matrix x);
}
