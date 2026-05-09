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
 * Box bounds {@code lb &le; x &le; ub} on the optimisation variable, plus
 * a global {@code keepFeasible} flag controlling whether the algorithm is
 * allowed to step outside the box during line-searches and trial steps.
 */
public final class Bounds {

	private final Matrix lb;
	private final Matrix ub;
	private final boolean keepFeasible;

	/**
	 * @param lb           per-variable lower bound ({@code n x 1});
	 *                     use {@link Double#NEGATIVE_INFINITY} to disable a bound
	 * @param ub           per-variable upper bound ({@code n x 1});
	 *                     use {@link Double#POSITIVE_INFINITY} to disable a bound
	 * @param keepFeasible if {@code true}, every trial iterate must stay
	 *                     within the bounds
	 */
	public Bounds(Matrix lb, Matrix ub, boolean keepFeasible) {
		// Defensive copy: the user could otherwise mutate `lb`/`ub` after
		// construction and silently change the constraint.
		this.lb = Matrix.Factory.copyFromMatrix(lb);
		this.ub = Matrix.Factory.copyFromMatrix(ub);
		this.keepFeasible = keepFeasible;
	}

	/** @return defensive copy of the lower bound ({@code n x 1}) */
	public Matrix lb() {
		return Matrix.Factory.copyFromMatrix(lb);
	}

	/** @return defensive copy of the upper bound ({@code n x 1}) */
	public Matrix ub() {
		return Matrix.Factory.copyFromMatrix(ub);
	}

	/** @return whether trial iterates must stay strictly inside the box */
	public boolean keepFeasible() {
		return keepFeasible;
	}

	/**
	 * Lower and upper residuals at {@code x}: {@code sl = x - lb} (positive
	 * iff lower bound satisfied), {@code sb = ub - x} (positive iff upper
	 * bound satisfied). Mirrors scipy's {@code Bounds.residual(x)}.
	 *
	 * @param x current iterate ({@code n x 1})
	 * @return {@code (sl, sb)} as a {@link Residual}
	 */
	public Residual residual(Matrix x) {
		long n = lb.getRowCount();
		double[] sl = new double[(int) n];
		double[] sb = new double[(int) n];
		for (int i = 0; i < n; ++i) {
			double xi = x.getAsDouble(i, 0);
			sl[i] = xi - lb.getAsDouble(i, 0);
			sb[i] = ub.getAsDouble(i, 0) - xi;
		}
		return new Residual(sl, sb);
	}
}
