package de.labathome;

import org.ujmp.core.Matrix;

public interface GradientAndJacobian {

	public Matrix grad(Matrix x);
	public Matrix jac(Matrix x);

}
