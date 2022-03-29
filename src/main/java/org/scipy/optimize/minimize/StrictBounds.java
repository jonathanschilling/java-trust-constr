package org.scipy.optimize.minimize;

import org.scipy.optimize.minimize.records.Bounds;

public class StrictBounds extends FiniteDifferenceBounds {

	public StrictBounds(Bounds bounds, long nVars) {
		super(bounds.lb(), bounds.ub(), bounds.keepFeasible(), nVars);
	}
}
