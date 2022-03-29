package org.scipy.optimize.minimize;

import org.ujmp.core.Matrix;

@FunctionalInterface
public interface IGradientAndJacobian {
	public GradientAndJacobian gradAndJac(Matrix z);
}
