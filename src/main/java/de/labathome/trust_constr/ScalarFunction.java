package de.labathome.trust_constr;

import org.ujmp.core.Matrix;

public class ScalarFunction {

	Function fun;
	Matrix x0;
	Object args;
	Gradient grad;
	Hessian hess;
	double finiteDiffRelStep;
	FiniteDifferenceBounds finiteDiffBounds;

	public ScalarFunction(Function fun, Matrix x0, Object args, Gradient grad, Hessian hess, double finiteDiffRelStep,
			FiniteDifferenceBounds finiteDiffBounds) {
		this.fun = fun;
		this.x0 = x0;
		this.args = args;
		this.grad = grad;
		this.hess = hess;
		this.finiteDiffRelStep = finiteDiffRelStep;
		this.finiteDiffBounds = finiteDiffBounds;
	}

	public int numEval() {
		return 0;
	}

	public int numGradientEval() {
		return 0;
	}

	public int numHessianEval() {
		return 0;
	}

	public double eval(Matrix x) {
		return Double.NaN;
	}

	public Matrix gradient(Matrix x) {
		return null;
	}

	public Matrix hessian(Matrix x) {
		return null;
	}

	public double lastEval() {
		return Double.NaN;
	}

	public Matrix lastGradient() {
		return null;
	}

	public Matrix lastHessian() {
		return null;
	}
}
