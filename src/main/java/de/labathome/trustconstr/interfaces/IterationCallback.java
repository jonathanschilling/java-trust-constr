package de.labathome.trustconstr.interfaces;

import de.labathome.trustconstr.records.State;

/**
 * Caller-supplied hook invoked once per outer iteration of the trust-constr
 * orchestrator. Mirrors scipy's {@code callback(xk, OptimizeResult-state)}
 * argument to {@code minimize(method='trust-constr', callback=...)}: the
 * caller observes intermediate state and returns {@code true} to request
 * early termination, otherwise {@code false} to continue.
 *
 * <p>The callback is fired after the inner stop-criterion has updated
 * {@link State#x}, {@link State#fun}, {@link State#grad}, {@link State#optimality},
 * {@link State#constrViolation}, {@link State#trustRadius}, and
 * {@link State#nIter} for the just-completed iteration. Returning
 * {@code true} terminates the SQP / IP loop the same way exhausting
 * {@code maxIter} does -- the orchestrator still populates
 * {@code OptimizeResult} from whatever state the loop reached.
 */
@FunctionalInterface
public interface IterationCallback {

	/**
	 * @param state outer-iteration state at the just-completed iteration
	 * @return {@code true} to request early termination, {@code false} to continue
	 */
	boolean shouldTerminate(State state);
}
