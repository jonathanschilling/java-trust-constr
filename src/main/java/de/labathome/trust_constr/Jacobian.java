package de.labathome.trust_constr;

import org.ujmp.core.Matrix;

public interface Jacobian {
	public Matrix jacEq(Matrix x);
	public Matrix jacIneq(Matrix x);
}
