package de.labathome.trustconstr.enums;

/**
 * What BFGS / SR1 should do when the curvature condition
 * {@code (delta_grad * delta_x) &le; threshold} is violated.
 */
public enum ExceptionStrategy {

	/** Skip the Hessian update entirely (scipy {@code 'skip_update'}). */
	SKIP_UPDATE,

	/** Damp the update by interpolating with the previous matrix
	 *  (scipy {@code 'damp_update'}). */
	DAMP_UPDATE;
}
