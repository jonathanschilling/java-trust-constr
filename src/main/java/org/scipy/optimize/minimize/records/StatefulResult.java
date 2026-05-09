package org.scipy.optimize.minimize.records;

import org.scipy.optimize.minimize.matrix.Matrix;

/**
 * Inner-loop result from {@link
 * org.scipy.optimize.minimize.EqualityConstrainedSQP#eqSQP} and
 * {@link org.scipy.optimize.minimize.TrustRegionInteriorPoint#trustRegionInteriorPoint}.
 * Carries the final iterate, the threaded-through {@link State}, and the
 * final Lagrange-multiplier vector.
 */
public final class StatefulResult {

	private Matrix x;
	private State state;
	private Matrix v;

	/**
	 * @param x     final iterate ({@code n x 1})
	 * @param state final outer-loop state
	 */
	public StatefulResult(Matrix x, State state) {
		this(x, state, null);
	}

	/**
	 * @param x     final iterate ({@code n x 1})
	 * @param state final outer-loop state
	 * @param v     final Lagrange multipliers (may be {@code null})
	 */
	public StatefulResult(Matrix x, State state, Matrix v) {
		this.x = x;
		this.state = state;
		this.v = v;
	}

	/** @return the final iterate */
	public Matrix x() {
		return x;
	}

	/** @return the final outer-loop state */
	public State state() {
		return state;
	}

	/**
	 * Final Lagrange multiplier vector from the inner SQP loop.
	 *
	 * <p>For pure-equality problems this is the equality multiplier, length
	 * {@code nEq}. For interior-point dispatch it is the augmented-system
	 * multiplier from the last barrier subproblem, length {@code nEq + nIneq}
	 * -- first {@code nEq} entries are equality multipliers, remaining
	 * {@code nIneq} entries are inequality (slack-row) multipliers
	 * corresponding to the original problem's lambda.
	 *
	 * @return final multiplier vector, or {@code null} if not populated
	 */
	public Matrix v() {
		return v;
	}
}
