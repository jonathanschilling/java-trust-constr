package org.scipy.optimize.minimize;

import org.ujmp.core.Matrix;

public interface VectorFunction {

	public int numEval();

	public int numGradientEval();

	public int numHessianEval();

	public boolean sparseJacobian();

	public Matrix eval(Matrix x);

	public Matrix gradient(Matrix x);

	public Matrix hessian(Matrix x, Matrix v);

	public Matrix lastEval();

	public Matrix lastGradient();

	public Matrix lastHessian();

	public Matrix lastV();
}
