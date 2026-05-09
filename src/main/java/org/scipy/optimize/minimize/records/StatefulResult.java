package org.scipy.optimize.minimize.records;

import org.ujmp.core.Matrix;

public final class StatefulResult {

	private Matrix x;
	private State state;
	private Matrix v;

	public StatefulResult(Matrix x, State state) {
		this(x, state, null);
	}

	public StatefulResult(Matrix x, State state, Matrix v) {
		this.x = x;
		this.state = state;
		this.v = v;
	}

	public Matrix x() {
		return x;
	}

	public State state() {
		return state;
	}

	/**
	 * Final Lagrange multiplier vector from the inner SQP loop.
	 *
	 * <p>For pure-equality problems this is the equality multiplier, length
	 * {@code nEq}. For interior-point dispatch it is the augmented-system
	 * multiplier from the last barrier subproblem, length {@code nEq + nIneq}
	 * — first {@code nEq} entries are equality multipliers, remaining
	 * {@code nIneq} entries are inequality (slack-row) multipliers
	 * corresponding to the original problem's λ.
	 *
	 * <p>May be {@code null} when not populated by the algorithm path.
	 */
	public Matrix v() {
		return v;
	}
}
