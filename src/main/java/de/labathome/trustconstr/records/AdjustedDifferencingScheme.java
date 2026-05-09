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
 * Output of {@link de.labathome.trustconstr.NumDiff#adjustSchemeToBounds}:
 * the per-dimension finite-difference step (with sign flips and clamping
 * applied so each step lands inside the user-provided bounds) plus a per-
 * dimension flag indicating whether the central scheme had to be downgraded
 * to a one-sided scheme.
 */
public final class AdjustedDifferencingScheme {

	private Matrix hAdjusted;
	private boolean[] useOneSided;

	/**
	 * @param hAdjusted   adjusted absolute step sizes ({@code n x 1})
	 * @param useOneSided per-dimension flag: {@code true} forces a one-sided
	 *                    scheme at this dimension, {@code false} keeps the
	 *                    requested two-sided scheme
	 */
	public AdjustedDifferencingScheme(Matrix hAdjusted, boolean[] useOneSided) {
		this.hAdjusted = hAdjusted;
		this.useOneSided = useOneSided;
	}

	/** @return the bound-adjusted step sizes ({@code n x 1}) */
	public Matrix hAdjusted() {
		return hAdjusted;
	}

	/** @return per-dimension one-sided/two-sided scheme flags */
	public boolean[] useOneSided() {
		return useOneSided;
	}
}
