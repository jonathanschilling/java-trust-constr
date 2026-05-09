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
package de.labathome.trustconstr.interfaces;

import de.labathome.trustconstr.matrix.Matrix;

/**
 * Hessian of the Lagrangian {@code L(x, v) = f(x) + Sum v_i c_i(x)} with
 * respect to {@code x}, returned as a {@link LinearOperator} so callers can
 * apply it to vectors without materialising the full {@code n &times; n}
 * matrix.
 */
@FunctionalInterface
public interface LagrangeHessian {

	/**
	 * @param x current iterate ({@code n &times; 1})
	 * @param v full Lagrange-multiplier vector (equality + inequality)
	 * @return {@link LinearOperator} applying {@code grad^2L(x, v)} to a vector
	 */
	public LinearOperator lagrHess(Matrix x, Matrix v);
}
