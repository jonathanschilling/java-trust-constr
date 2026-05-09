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
package de.labathome.optimization;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.HessianLinearOperator;
import de.labathome.trustconstr.interfaces.HessianProduct;
import de.labathome.trustconstr.matrix.Matrix;


class TestHessianLinearOperator {

	private static final double TOL = 1.0e-14;

	@Test
	void testIdentityHessian() {
		// hessp(x, p) = p  ->  the wrapper should materialise H = I_n.
		HessianProduct hessp = (x, p, args) -> p;
		HessianLinearOperator op = new HessianLinearOperator(hessp, 3);
		Matrix h = op.apply(Matrix.Factory.linkToArray(new double[] {0.0, 0.0, 0.0}), null);
		Assertions.assertEquals(3, h.getRowCount());
		Assertions.assertEquals(3, h.getColumnCount());
		double[][] expected = { {1, 0, 0}, {0, 1, 0}, {0, 0, 1} };
		RelAbsAssertions.assertArrayRelAbsEquals(expected, h.toDoubleArray(), TOL);
	}

	@Test
	void testKnownDenseHessian() {
		// H = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]   (1D Laplacian)
		final double[][] hData = {
				{ 2, -1,  0 },
				{ -1, 2, -1 },
				{ 0, -1,  2 },
		};
		HessianProduct hessp = (x, p, args) -> {
			double[] r = new double[3];
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					r[i] += hData[i][j] * p.getAsDouble(j, 0);
				}
			}
			return Matrix.Factory.linkToArray(r);
		};
		HessianLinearOperator op = new HessianLinearOperator(hessp, 3);
		Matrix h = op.apply(Matrix.Factory.linkToArray(new double[3]), null);
		RelAbsAssertions.assertArrayRelAbsEquals(hData, h.toDoubleArray(), TOL);
	}
}
