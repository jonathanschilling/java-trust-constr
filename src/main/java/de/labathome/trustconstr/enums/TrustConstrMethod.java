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
 * Which dispatch path actually ran inside
 * {@link de.labathome.trustconstr.MinimizeTrustConstr}: reported back to
 * the caller via {@link de.labathome.trustconstr.records.OptimizeResult#method}.
 */
public enum TrustConstrMethod {

	/** Trust-region equality-constrained SQP (scipy {@code equality_constrained_sqp}). */
	EQUALITY_CONSTRAINED_SQP,

	/** Trust-region interior-point method (scipy {@code tr_interior_point}). */
	TRUST_REGION_INTERIOR_POINT;
}
