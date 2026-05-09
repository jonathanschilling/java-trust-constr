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
 * What BFGS / SR1 should do when the curvature condition
 * {@code (delta_grad * delta_x) &le; threshold} is violated.
 */
public enum ExceptionStrategy {

	/** Skip the Hessian update entirely (scipy {@code 'skip_update'}). */
	SKIP_UPDATE,

	/** Damp the update by interpolating with the previous matrix
	 *  (scipy {@code 'damp_update'}). */
	DAMP_UPDATE;
}
