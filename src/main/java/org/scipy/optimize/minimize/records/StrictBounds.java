package org.scipy.optimize.minimize.records;

public class StrictBounds extends FiniteDifferenceBounds {

	public StrictBounds(Bounds bounds) {
		super(bounds.lb(), bounds.ub(), bounds.keepFeasible());
	}
}
