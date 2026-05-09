/*
 * Copyright 2026 Jonathan Schilling
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package de.labathome.trustconstr.records;

import de.labathome.trustconstr.matrix.Matrix;

/**
 * Inner-loop result from {@link
 * de.labathome.trustconstr.EqualityConstrainedSQP#eqSQP} and
 * {@link de.labathome.trustconstr.TrustRegionInteriorPoint#trustRegionInteriorPoint}.
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
