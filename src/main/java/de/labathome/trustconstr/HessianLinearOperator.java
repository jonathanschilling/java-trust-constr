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
package de.labathome.trustconstr;

import java.util.function.BiFunction;

import de.labathome.trustconstr.interfaces.HessianProduct;
import de.labathome.trustconstr.matrix.Matrix;

/**
 * Adapter that exposes a matrix-free {@link HessianProduct} as a
 * {@link BiFunction} returning an explicit Hessian {@link Matrix}.
 *
 * <p>Mirrors scipy's {@code LinearOperator}-wrapping of {@code hessp} in
 * {@code _minimize_trustregion_constr} ({@code minimize_trustregion_constr.py:36}):
 * given a callable {@code hessp(x, p)} returning {@code H(x)*p} for each
 * vector {@code p}, the wrapper materialises {@code H(x)} as a dense
 * {@code n x n} matrix by probing {@code hessp} along each Cartesian basis
 * vector. This is O(n) hessp evaluations -- sufficient for the modest
 * problem sizes trust-constr is typically used on, and a future hook for
 * a true matrix-free path when downstream consumers can accept one.
 */
public class HessianLinearOperator implements BiFunction<Matrix, Object, Matrix> {

	private final HessianProduct hessp;
	private final long nVars;

	/**
	 * @param hessp matrix-free Hessian-vector product evaluator
	 *              {@code (x, p, args) -> H(x)*p}
	 * @param nVars number of decision variables (= dimension of the Hessian)
	 */
	public HessianLinearOperator(HessianProduct hessp, long nVars) {
		this.hessp = hessp;
		this.nVars = nVars;
	}

	@Override
	public Matrix apply(Matrix x, Object args) {
		int n = (int) nVars;
		double[][] hDense = new double[n][n];
		Matrix p = Matrix.Factory.zeros(n, 1);
		for (int j = 0; j < n; ++j) {
			if (j > 0) {
				p.setAsDouble(0.0, j - 1, 0);
			}
			p.setAsDouble(1.0, j, 0);
			Matrix hp = hessp.apply(x, p, args);
			for (int i = 0; i < n; ++i) {
				hDense[i][j] = hp.getAsDouble(i, 0);
			}
		}
		return Matrix.Factory.linkToArray(hDense);
	}
}
