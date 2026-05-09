package de.labathome.trustconstr.enums;

/**
 * Finite-difference scheme used by
 * {@link de.labathome.trustconstr.NumDiff#approxDerivative}.
 */
public enum FiniteDifferenceMethod {

	/** Forward difference (scipy {@code '2-point'}). */
	TWO_POINT,

	/** One-sided perturbation (forward or backward, picked per dimension). */
	ONE_SIDED,

	/** Central difference (scipy {@code '3-point'}). */
	THREE_POINT,

	/** Two-sided central perturbation. */
	TWO_SIDED,

	/** Complex-step differentiation (scipy {@code 'cs'}). Not yet implemented. */
	COMPLEX_STEP;
}
