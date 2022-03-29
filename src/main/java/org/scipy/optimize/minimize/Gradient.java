package org.scipy.optimize.minimize;

import org.ujmp.core.Matrix;

public interface Gradient {
	public Matrix grad(Matrix x, Object args);
}
