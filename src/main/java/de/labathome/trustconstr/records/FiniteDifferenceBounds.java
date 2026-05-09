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
 * Box bounds {@code lb &le; x &le; ub} used by
 * {@link de.labathome.trustconstr.NumDiff#approxDerivative} to clamp
 * finite-difference perturbations.
 */
public class FiniteDifferenceBounds {

	private final Matrix lb;
	private final Matrix ub;
	private final boolean keepFeasible;

	/**
	 * @param lb           per-variable lower bound ({@code n &times; 1})
	 * @param ub           per-variable upper bound ({@code n &times; 1})
	 * @param keepFeasible if {@code true}, every FD perturbation must stay
	 *                     strictly inside the box
	 */
	public FiniteDifferenceBounds(Matrix lb, Matrix ub, boolean keepFeasible) {
		// Defensive copy so mutating the input after construction can't
		// silently change the FD bounds.
		this.lb = Matrix.Factory.copyFromMatrix(lb);
		this.ub = Matrix.Factory.copyFromMatrix(ub);
		this.keepFeasible = keepFeasible;
	}

	/**
	 * Convenience factory for the bound-free case: {@code -inf &le; x &le; +inf}.
	 *
	 * @param nVars number of decision variables
	 * @return an unbounded {@link FiniteDifferenceBounds} of the requested size
	 */
	public static FiniteDifferenceBounds unbounded(long nVars) {
		Matrix lb = Matrix.Factory.ones(nVars, 1).times(Double.NEGATIVE_INFINITY);
		Matrix ub = Matrix.Factory.ones(nVars, 1).times(Double.POSITIVE_INFINITY);
		return new FiniteDifferenceBounds(lb, ub, false);
	}

	/** @return defensive copy of the lower bounds ({@code n &times; 1}) */
	public Matrix lb() {
		return Matrix.Factory.copyFromMatrix(lb);
	}

	/** @return defensive copy of the upper bounds ({@code n &times; 1}) */
	public Matrix ub() {
		return Matrix.Factory.copyFromMatrix(ub);
	}

	/** @return whether perturbations must stay strictly inside the box */
	public boolean keepFeasible() {
		return keepFeasible;
	}
}
