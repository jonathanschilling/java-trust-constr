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
package de.labathome.trustconstr.records;

import de.labathome.trustconstr.matrix.Matrix;

/** Bundle of a gradient {@code gradf(x)} and a Jacobian {@code dc/dx}. */
public final class GradientAndJacobian {

	private Matrix grad;
	private Matrix jac;

	/**
	 * @param grad objective gradient ({@code n &times; 1})
	 * @param jac  constraint Jacobian ({@code m &times; n})
	 */
	public GradientAndJacobian(Matrix grad, Matrix jac) {
		this.grad = grad;
		this.jac = jac;
	}

	/** @return objective gradient ({@code n &times; 1}) */
	public Matrix grad() {
		return grad;
	}

	/** @return constraint Jacobian ({@code m &times; n}) */
	public Matrix jac() {
		return jac;
	}
}
