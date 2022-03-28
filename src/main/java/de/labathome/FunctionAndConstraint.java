package de.labathome;

import org.ujmp.core.Matrix;

public interface FunctionAndConstraint {
	public double objective(Matrix x);
	public Matrix constraint(Matrix x);
}
