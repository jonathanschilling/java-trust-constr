package org.scipy.optimize.minimize;

import org.ujmp.core.Matrix;

@FunctionalInterface
public interface Hessian {
	public Matrix hess(Matrix x, Object args);
}
