package de.labathome.trustconstr.enums;

/**
 * Which dispatch path actually ran inside
 * {@link de.labathome.trustconstr.MinimizeTrustConstr}: reported back to
 * the caller via {@link de.labathome.trustconstr.records.OptimizeResult#method}.
 */
public enum TrustConstrMethod {

	/** Trust-region equality-constrained SQP (scipy {@code equality_constrained_sqp}). */
	EQUALITY_CONSTRAINED_SQP,

	/** Trust-region interior-point method (scipy {@code tr_interior_point}). */
	TRUST_REGION_INTERIOR_POINT;
}
