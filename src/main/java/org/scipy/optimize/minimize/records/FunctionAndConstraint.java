package org.scipy.optimize.minimize.records;

import org.scipy.optimize.minimize.matrix.Matrix;

/** Bundle of an objective value {@code f(x)} and a constraint vector {@code c(x)}. */
public final class FunctionAndConstraint {

	private double f;
	private Matrix c;

	/**
	 * @param f scalar objective value
	 * @param c constraint vector
	 */
	public FunctionAndConstraint(double f, Matrix c) {
		this.f = f;
		this.c = c;
	}

	/** @return objective value {@code f(x)} */
	public double f() {
		return f;
	}

	/** @return constraint vector {@code c(x)} */
	public Matrix c() {
		return c;
	}
}
