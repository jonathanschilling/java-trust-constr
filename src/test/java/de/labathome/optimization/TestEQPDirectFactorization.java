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

import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.QPSubproblem;
import de.labathome.trustconstr.matrix.Matrix;


class TestEQPDirectFactorization {

	/**
	 * Example 16.2 in Nocedal/Wright, "Numerical Optimization" (2006), p. 452
	 */
	@Test
	void testNocedalExample() {
		final double tolerance = 1.0e-15;

		Matrix H = Matrix.Factory.importFromArray(new double[][] {
				{ 6.0, 2.0, 1.0 },
				{ 2.0, 5.0, 2.0 },
				{ 1.0, 2.0, 4.0 }
		});

		Matrix A = Matrix.Factory.importFromArray(new double[][] {
				{ 1.0, 0.0, 1.0 },
				{ 0.0, 1.0, 1.0 }
		});

		Matrix c = Matrix.Factory.importFromArray(new double[][] {
				{-8.0},
				{-3.0},
				{-3.0}
		});

		// This is actually -b.
		Matrix b = Matrix.Factory.importFromArray(new double[][] {
				{-3.0},
				{ 0.0}
		});

		Matrix[] xLambda = QPSubproblem.eqpKktFact(H, c, A, b);
		final double[] x = xLambda[0].toColumnArray();
		final double[] lambda = xLambda[1].toColumnArray();

		final double[] expectedX = { 2.0, -1.0, 1.0 };
		final double[] expectedLambda = { 3.0, -2.0 };
		RelAbsAssertions.assertArrayRelAbsEquals(expectedX, x, tolerance);
		RelAbsAssertions.assertArrayRelAbsEquals(expectedLambda, lambda, tolerance);
	}

}
