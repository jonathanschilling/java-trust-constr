package org.scipy.optimize.minimize;

public class StrictBounds extends FiniteDifferenceBounds {

	public StrictBounds(Bounds bounds, long nVars) {
		super(bounds.lb(), bounds.ub(), bounds.keepFeasible(), nVars);
	}
}
