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

import java.util.List;

import de.labathome.trustconstr.enums.PCGStoppingCondition;
import de.labathome.trustconstr.matrix.Matrix;

/**
 * Result of a projected-CG inner solve from
 * {@link de.labathome.trustconstr.QPSubproblem#projectedCG}. Carries the
 * solution vector, iteration count, why CG terminated, optionally the
 * full iterate trajectory, and a flag indicating whether the proposed step
 * landed on the trust-region boundary.
 */
public class CGInfo {

	/** Solution vector of the EQP problem ({@code n x 1}). */
	public Matrix x;

	/** Number of CG iterations performed. */
	public int niter;

	/** Why CG terminated (residual met tolerance, hit boundary, etc.). */
	public PCGStoppingCondition stopCond;

	/** All intermediate iterates, when {@code returnAll=true}; otherwise {@code null}. */
	public List<Matrix> allVecs;

	/** {@code true} iff the returned step lies on the trust-region boundary. */
	public boolean hitsBoundary;
}
