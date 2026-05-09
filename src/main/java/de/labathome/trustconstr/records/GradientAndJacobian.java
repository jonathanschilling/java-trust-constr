package de.labathome.trustconstr.records;

import de.labathome.trustconstr.matrix.Matrix;

/** Bundle of a gradient {@code gradf(x)} and a Jacobian {@code dc/dx}. */
public final class GradientAndJacobian {

	private Matrix grad;
	private Matrix jac;

	/**
	 * @param grad objective gradient ({@code n &times; 1})
	 * @param jac  constraint Jacobian ({@code m &times; n})
	 */
	public GradientAndJacobian(Matrix grad, Matrix jac) {
		this.grad = grad;
		this.jac = jac;
	}

	/** @return objective gradient ({@code n &times; 1}) */
	public Matrix grad() {
		return grad;
	}

	/** @return constraint Jacobian ({@code m &times; n}) */
	public Matrix jac() {
		return jac;
	}
}
