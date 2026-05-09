package org.scipy.optimize.minimize.records;

import org.scipy.optimize.minimize.matrix.Matrix;

public class FiniteDifferenceBounds {

	private Matrix lb;
	private Matrix ub;
	private boolean keepFeasible;

	public FiniteDifferenceBounds(Matrix lb, Matrix ub, boolean keepFeasible) {
		this.lb = lb;
		this.ub = ub;
		this.keepFeasible = keepFeasible;
	}

	public static FiniteDifferenceBounds unbounded(long nVars) {
		Matrix lb = Matrix.Factory.ones(nVars, 1).times(Double.NEGATIVE_INFINITY);
		Matrix ub = Matrix.Factory.ones(nVars, 1).times(Double.POSITIVE_INFINITY);
		return new FiniteDifferenceBounds(lb, ub, false);
	}

	public Matrix lb() {
		return lb;
	}

	public Matrix ub() {
		return ub;
	}

	public boolean keepFeasible() {
		return keepFeasible;
	}
}
