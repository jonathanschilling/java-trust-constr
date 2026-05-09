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
 * How {@link de.labathome.trustconstr.Projections#projections} should
 * factor the constraint Jacobian to build the null-space / least-squares /
 * row-space projectors.
 */
public enum ProjectionMethod {

	/**
	 * The operators will be computed using the so-called normal equation approach
	 * explained in [1]_. In order to do so the Cholesky factorization of
	 * {@code (A A.T)} is computed. Exclusive for sparse matrices.
	 */
	NORMAL_EQUATION,

	/**
	 * The operators will be computed using the so-called augmented system approach
	 * explained in [1]_. Exclusive for sparse matrices.
	 */
	AUGMENTED_SYSTEM,

	/**
	 * Compute projections using QR factorization.
	 * Exclusive for dense matrices.
	 */
	QR_FACTORIZATION,

	/**
	 * Compute projections using SVD factorization.
	 * Exclusive for dense matrices.
	 */
	SVD_FACTORIZATION;
}
