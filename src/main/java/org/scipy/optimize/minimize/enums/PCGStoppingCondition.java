package org.scipy.optimize.minimize.enums;

/** Why projected-CG terminated on the most recent inner solve. */
public enum PCGStoppingCondition {

	/** {@code 0}: CG subproblem not evaluated. */
	NOT_EVALUATED,

	/** {@code 1}: Iteration limit was reached. */
	ITER_LIMIT_REACHED,

	/** {@code 2}: Reached the trust-region boundary. */
	TRUST_REGION_BOUNDARY_REACHED,

	/** {@code 3}: Negative curvature detected. */
	NEGATIVE_CURVATURE,

	/** {@code 4}: Tolerance was satisfied. */
	TOLERANCE_SATISFIED
}
