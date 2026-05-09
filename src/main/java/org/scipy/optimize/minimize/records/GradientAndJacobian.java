package org.scipy.optimize.minimize.records;

import org.scipy.optimize.minimize.matrix.Matrix;

public final class GradientAndJacobian {

	private Matrix grad;
	private Matrix jac;

	public GradientAndJacobian(Matrix grad, Matrix jac) {
		this.grad = grad;
		this.jac = jac;
	}

	public Matrix grad() {
		return grad;
	}

	public Matrix jac() {
		return jac;
	}
}
