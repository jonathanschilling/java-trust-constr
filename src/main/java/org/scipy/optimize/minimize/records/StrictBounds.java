package org.scipy.optimize.minimize.records;

/**
 * {@link FiniteDifferenceBounds} variant constructed from a {@link Bounds}
 * record, propagating the {@code keep_feasible} flag. Used by the
 * full-shape {@code minimize} adapter so finite-difference gradient
 * perturbations honour the user's {@code Bounds(keep_feasible=True)}.
 */
public class StrictBounds extends FiniteDifferenceBounds {

	/**
	 * @param bounds source bounds (lb / ub / keepFeasible) to wrap
	 */
	public StrictBounds(Bounds bounds) {
		super(bounds.lb(), bounds.ub(), bounds.keepFeasible());
	}
}
