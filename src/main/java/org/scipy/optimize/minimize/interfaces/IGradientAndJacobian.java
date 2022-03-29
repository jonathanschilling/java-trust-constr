package org.scipy.optimize.minimize.interfaces;

import org.scipy.optimize.minimize.GradientAndJacobian;
import org.ujmp.core.Matrix;

@FunctionalInterface
public interface IGradientAndJacobian {
	public GradientAndJacobian gradAndJac(Matrix z);
}
