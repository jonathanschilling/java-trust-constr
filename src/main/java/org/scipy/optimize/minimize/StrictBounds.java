package org.scipy.optimize.minimize;

public class StrictBounds extends FiniteDifferenceBounds {

	public StrictBounds(double[] lb, double[] ub, boolean keepFeasible, long nVars) {
		super(lb, ub, keepFeasible, nVars);
	}
}
