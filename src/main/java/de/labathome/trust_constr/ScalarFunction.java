package de.labathome.trust_constr;

import org.ujmp.core.Matrix;

public interface ScalarFunction {
	public int numEval();
	public int numGradientEval();
	public int numHessianEval();

	public double eval(Matrix x);
	public Matrix gradient(Matrix x);
	public Matrix hessian(Matrix x);

	public double lastEval();
	public Matrix lastGradient();
	public Matrix lastHessian();
}
