package org.scipy.optimize.minimize.interfaces;

import org.scipy.optimize.minimize.matrix.Matrix;

public interface Jacobian {
	public Matrix jacEq(Matrix x);

	public Matrix jacIneq(Matrix x);
}
