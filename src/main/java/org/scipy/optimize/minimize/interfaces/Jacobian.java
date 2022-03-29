package org.scipy.optimize.minimize.interfaces;

import org.ujmp.core.Matrix;

public interface Jacobian {
	public Matrix jacEq(Matrix x);

	public Matrix jacIneq(Matrix x);
}
