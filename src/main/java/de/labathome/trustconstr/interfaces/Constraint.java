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
 * Evaluator for the canonical-form constraints of the optimization problem:
 * {@code constr_eq(x) = 0} and {@code constr_ineq(x) &le; 0}.
 */
public interface Constraint {

	/**
	 * @param x current iterate ({@code n &times; 1})
	 * @return equality-constraint residual ({@code nEq &times; 1})
	 */
	public Matrix constrEq(Matrix x);

	/**
	 * @param x current iterate ({@code n &times; 1})
	 * @return inequality-constraint residual ({@code nIneq &times; 1});
	 *         feasible iff every entry is {@code &le; 0}
	 */
	public Matrix constrIneq(Matrix x);
}
