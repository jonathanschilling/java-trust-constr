package de.labathome.trust_constr;

public enum PCGStoppingCondition {
	/** 0: CG subproblem not evaluated */
	NOT_EVALUATED,

	/** 1: Iteration limit was reached */
	ITER_LIMIT_REACHED,

	/** 2: Reached the trust-region boundary */
	TRUST_REGION_BOUNDARY_REACHED,

	/** 3: Negative curvature detected */
	NEGATIVE_CURVATURE,

	/** 4: Tolerance was satisfied */
	TOLERANCE_SATISFIED
}
