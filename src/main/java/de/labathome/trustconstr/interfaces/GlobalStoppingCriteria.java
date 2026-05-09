package de.labathome.trustconstr.interfaces;

import de.labathome.trustconstr.records.CGInfo;
import de.labathome.trustconstr.records.State;
import de.labathome.trustconstr.matrix.Matrix;

/**
 * Termination predicate consulted by the trust-region SQP and interior-point
 * loops once per outer iteration. Implementations decide whether the
 * algorithm has converged based on the current optimality and constraint-
 * violation measures, the trust-region radius, the merit-function penalty,
 * and IP-specific quantities (barrier parameter, inner-loop tolerance).
 */
public interface GlobalStoppingCriteria {

	/**
	 * @param state current outer-iteration state
	 * @param x     current iterate
	 * @param lastIterationFailed whether the previous trial step was rejected
	 * @param optimality KKT optimality measure (infinity-norm of Lagrangian gradient)
	 * @param constrViolation infinity-norm of the constraint residual
	 * @param trustRadius current trust-region radius
	 * @param penalty current merit-function penalty
	 * @param cgInfo info from the projected-CG inner solve
	 * @param barrierParameter IP-only: current log-barrier coefficient
	 * @param tolerance IP-only: current barrier-subproblem tolerance
	 * @return {@code true} iff the outer loop should terminate
	 */
	public boolean shouldStop(State state, Matrix x, boolean lastIterationFailed, double optimality,
			double constrViolation, double trustRadius, double penalty, CGInfo cgInfo, double barrierParameter,
			double tolerance);
}
