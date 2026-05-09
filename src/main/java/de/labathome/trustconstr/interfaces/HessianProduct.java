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
 * A matrix-free Hessian: given the current point {@code x} and a vector
 * {@code p}, return {@code H(x) p}.
 *
 * Mirrors the {@code hessp(x, p, *args)} callable from scipy's
 * {@code minimize(method='trust-constr')} API. Pairs with
 * {@link de.labathome.trustconstr.HessianLinearOperator}, which adapts a
 * {@code HessianProduct} to the explicit-matrix slot used elsewhere in the
 * port.
 */
@FunctionalInterface
public interface HessianProduct {

	/**
	 * @param x    [n] current point
	 * @param p    [n] vector to multiply
	 * @param args optional extra arguments forwarded by the caller
	 * @return     [n] result of {@code H(x) p}
	 */
	Matrix apply(Matrix x, Matrix p, Object args);
}
