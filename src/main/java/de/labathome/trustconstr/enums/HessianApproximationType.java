package de.labathome.trustconstr.enums;

/**
 * Whether a Hessian-update strategy approximates the Hessian or its inverse.
 */
public enum HessianApproximationType {

	/** Approximate the Hessian {@code B ~= grad^2f} (scipy {@code 'hess'}). */
	HESSIAN,

	/** Approximate the inverse Hessian {@code H ~= (grad^2f)^-^1}
	 *  (scipy {@code 'inv_hess'}). */
	INV_HESSIAN;
}
