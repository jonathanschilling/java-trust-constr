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
 * Whether a Hessian-update strategy approximates the Hessian or its inverse.
 */
public enum HessianApproximationType {

	/** Approximate the Hessian {@code B ~= grad^2f} (scipy {@code 'hess'}). */
	HESSIAN,

	/** Approximate the inverse Hessian {@code H ~= (grad^2f)^-^1}
	 *  (scipy {@code 'inv_hess'}). */
	INV_HESSIAN;
}
