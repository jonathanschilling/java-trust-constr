package org.scipy.optimize.minimize.interfaces;

import org.ujmp.core.Matrix;

@FunctionalInterface
public interface Function {

	/**
	 * Evaluates the scalar function.
	 *
	 * @param x    [n] argument
	 * @param args any additional fixed parameters needed to completely specify the
	 *             function
	 * @return scalar value of the function
	 */
	public double fun(Matrix x, Object args);
}
