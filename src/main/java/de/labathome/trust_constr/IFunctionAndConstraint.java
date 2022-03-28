package de.labathome.trust_constr;

import org.ujmp.core.Matrix;

@FunctionalInterface
public interface IFunctionAndConstraint {
	public FunctionAndConstraint funAndConstr(Matrix x);
}
