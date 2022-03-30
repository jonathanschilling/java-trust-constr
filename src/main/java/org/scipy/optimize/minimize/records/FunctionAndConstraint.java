package org.scipy.optimize.minimize.records;

import org.ujmp.core.Matrix;

public final class FunctionAndConstraint {

	private double f;
	private Matrix c;

	public FunctionAndConstraint(double f, Matrix c) {
		this.f = f;
		this.c = c;
	}

	public double f() {
		return f;
	}

	public Matrix c() {
		return c;
	}
}
