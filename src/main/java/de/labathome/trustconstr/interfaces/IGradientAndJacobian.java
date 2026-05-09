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

import de.labathome.trustconstr.records.GradientAndJacobian;
import de.labathome.trustconstr.matrix.Matrix;

/** Combined evaluator returning {@code (gradf(x), gradconstr(x))} in a single call. */
@FunctionalInterface
public interface IGradientAndJacobian {

	/**
	 * @param z current iterate
	 * @return gradient and Jacobian at {@code z}
	 */
	public GradientAndJacobian gradAndJac(Matrix z);
}
