package org.scipy.optimize.minimize.interfaces;

import org.ujmp.core.Matrix;

@FunctionalInterface
public interface Gradient {
	public Matrix grad(Matrix x, Object args);
}
