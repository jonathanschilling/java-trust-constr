package de.labathome.trust_constr;

import org.ujmp.core.Matrix;

@FunctionalInterface
public interface IGradientAndJacobian {
	public GradientAndJacobian gradAndJac(Matrix z);
}
