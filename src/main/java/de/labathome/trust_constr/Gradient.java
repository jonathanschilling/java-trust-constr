package de.labathome.trust_constr;

import org.ujmp.core.Matrix;

public interface Gradient {
	public Matrix grad(Matrix x);
}
