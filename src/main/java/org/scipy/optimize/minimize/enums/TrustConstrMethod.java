package org.scipy.optimize.minimize.enums;

/**
 * Which dispatch path actually ran inside
 * {@link org.scipy.optimize.minimize.MinimizeTrustConstr}: reported back to
 * the caller via {@link org.scipy.optimize.minimize.records.OptimizeResult#method}.
 */
public enum TrustConstrMethod {

	/** Trust-region equality-constrained SQP (scipy {@code equality_constrained_sqp}). */
	EQUALITY_CONSTRAINED_SQP,

	/** Trust-region interior-point method (scipy {@code tr_interior_point}). */
	TRUST_REGION_INTERIOR_POINT;
}
