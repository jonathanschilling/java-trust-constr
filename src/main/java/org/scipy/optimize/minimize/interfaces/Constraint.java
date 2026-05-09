package org.scipy.optimize.minimize.interfaces;

import org.scipy.optimize.minimize.matrix.Matrix;

public interface Constraint {
	public Matrix constrEq(Matrix x);

	public Matrix constrIneq(Matrix x);
}
