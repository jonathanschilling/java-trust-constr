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
 * Jacobian evaluator paired with {@link Constraint}: returns the partial
 * derivatives of the canonical-form equality and inequality constraint
 * vectors with respect to {@code x}.
 */
public interface Jacobian {

	/**
	 * @param x current iterate ({@code n &times; 1})
	 * @return Jacobian of {@code constr_eq} at {@code x} ({@code nEq &times; n})
	 */
	public Matrix jacEq(Matrix x);

	/**
	 * @param x current iterate ({@code n &times; 1})
	 * @return Jacobian of {@code constr_ineq} at {@code x}
	 *         ({@code nIneq &times; n})
	 */
	public Matrix jacIneq(Matrix x);
}
