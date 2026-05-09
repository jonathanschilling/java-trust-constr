package org.scipy.optimize.minimize.interfaces;

import org.scipy.optimize.minimize.records.GradientAndJacobian;
import org.scipy.optimize.minimize.matrix.Matrix;

@FunctionalInterface
public interface IGradientAndJacobian {
	public GradientAndJacobian gradAndJac(Matrix z);
}
