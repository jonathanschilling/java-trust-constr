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

/**
 * Finite-difference scheme used by
 * {@link de.labathome.trustconstr.NumDiff#approxDerivative}.
 */
public enum FiniteDifferenceMethod {

	/** Forward difference (scipy {@code '2-point'}). */
	TWO_POINT,

	/** One-sided perturbation (forward or backward, picked per dimension). */
	ONE_SIDED,

	/** Central difference (scipy {@code '3-point'}). */
	THREE_POINT,

	/** Two-sided central perturbation. */
	TWO_SIDED,

	/** Complex-step differentiation (scipy {@code 'cs'}). Not yet implemented. */
	COMPLEX_STEP;
}
