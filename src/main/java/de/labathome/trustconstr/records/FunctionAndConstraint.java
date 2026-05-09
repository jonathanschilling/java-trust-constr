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

/** Bundle of an objective value {@code f(x)} and a constraint vector {@code c(x)}. */
public final class FunctionAndConstraint {

	private double f;
	private Matrix c;

	/**
	 * @param f scalar objective value
	 * @param c constraint vector
	 */
	public FunctionAndConstraint(double f, Matrix c) {
		this.f = f;
		this.c = c;
	}

	/** @return objective value {@code f(x)} */
	public double f() {
		return f;
	}

	/** @return constraint vector {@code c(x)} */
	public Matrix c() {
		return c;
	}
}
