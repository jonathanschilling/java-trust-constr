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
package de.labathome.trustconstr.enums;

/** Why projected-CG terminated on the most recent inner solve. */
public enum PCGStoppingCondition {

	/** {@code 0}: CG subproblem not evaluated. */
	NOT_EVALUATED,

	/** {@code 1}: Iteration limit was reached. */
	ITER_LIMIT_REACHED,

	/** {@code 2}: Reached the trust-region boundary. */
	TRUST_REGION_BOUNDARY_REACHED,

	/** {@code 3}: Negative curvature detected. */
	NEGATIVE_CURVATURE,

	/** {@code 4}: Tolerance was satisfied. */
	TOLERANCE_SATISFIED
}
