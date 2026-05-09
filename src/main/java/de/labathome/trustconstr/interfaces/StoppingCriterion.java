package de.labathome.trustconstr.interfaces;

import de.labathome.trustconstr.records.CGInfo;
import de.labathome.trustconstr.records.State;
import de.labathome.trustconstr.matrix.Matrix;

/**
 * Termination predicate consulted by the inner SQP loop. Lighter-weight
 * counterpart to {@link GlobalStoppingCriteria} for code paths that don't
 * need the barrier-related quantities.
 */
public interface StoppingCriterion {

	/**
	 * @param state current outer-iteration state
	 * @param x     current iterate
	 * @param lastIterationFailed whether the previous trial step was rejected
	 * @param optimality KKT optimality measure
	 * @param constrViolation infinity-norm of the constraint residual
	 * @param trustRadius current trust-region radius
	 * @param penalty current merit-function penalty
	 * @param cgInfo info from the projected-CG inner solve
	 * @return {@code true} iff the inner loop should terminate
	 */
	public boolean shouldStop(State state, Matrix x, boolean lastIterationFailed, double optimality,
			double constrViolation, double trustRadius, double penalty, CGInfo cgInfo);
}
