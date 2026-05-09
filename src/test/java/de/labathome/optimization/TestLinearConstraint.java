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
import de.labathome.trustconstr.matrix.MatrixOps;
import de.labathome.trustconstr.LinearConstraint;
import de.labathome.trustconstr.matrix.Matrix;


class TestLinearConstraint {

	private static final double TOL = 1.0e-14;
	private static final double NEG_INF = Double.NEGATIVE_INFINITY;
	private static final double POS_INF = Double.POSITIVE_INFINITY;

	@Test
	void testEqualityOnly() {
		// A = [[1, 2], [3, 4]], lb == ub == [5, 6] (pure equality)
		Matrix A = Matrix.Factory.linkToArray(new double[][] { {1, 2}, {3, 4} });
		LinearConstraint c = new LinearConstraint(A, new double[] {5, 6}, new double[] {5, 6});
		Assertions.assertEquals(2, c.nEq());
		Assertions.assertEquals(0, c.nIneq());

		Matrix x = Matrix.Factory.linkToArray(new double[] {1.0, 2.0});
		// A x = [1+4, 3+8] = [5, 11];  constrEq = A x - lb = [0, 5]
		RelAbsAssertions.assertArrayRelAbsEquals(new double[] {0.0, 5.0}, c.constrEq(x).toColumnArray(), TOL);
		RelAbsAssertions.assertArrayRelAbsEquals(new double[][] { {1, 2}, {3, 4} }, c.jacEq(x).toDoubleArray(), TOL);
	}

	@Test
	void testOneSidedUpper() {
		// A x <= [3, 7]
		Matrix A = Matrix.Factory.linkToArray(new double[][] { {1, 0}, {0, 1} });
		LinearConstraint c = new LinearConstraint(A, new double[] {NEG_INF, NEG_INF}, new double[] {3, 7});
		Assertions.assertEquals(0, c.nEq());
		Assertions.assertEquals(2, c.nIneq());

		Matrix x = Matrix.Factory.linkToArray(new double[] {2.0, 9.0});
		// constrIneq[i] = +1 * (A[i] x - ub[i]) = [-1, 2]
		RelAbsAssertions.assertArrayRelAbsEquals(new double[] {-1.0, 2.0}, c.constrIneq(x).toColumnArray(), TOL);
		// jacIneq = +1 * A
		RelAbsAssertions.assertArrayRelAbsEquals(new double[][] { {1, 0}, {0, 1} }, c.jacIneq(x).toDoubleArray(), TOL);
	}

	@Test
	void testOneSidedLower() {
		// A x >= [1, 1]   ==   -(A x) <= -[1,1]   ==   constrIneq = -(A x - lb) = lb - A x
		Matrix A = Matrix.Factory.linkToArray(new double[][] { {1, 0}, {0, 1} });
		LinearConstraint c = new LinearConstraint(A, new double[] {1, 1}, new double[] {POS_INF, POS_INF});
		Assertions.assertEquals(0, c.nEq());
		Assertions.assertEquals(2, c.nIneq());

		Matrix x = Matrix.Factory.linkToArray(new double[] {2.0, 0.5});
		// constrIneq[i] = -1 * (A[i] x - lb[i]) = [-(2-1), -(0.5-1)] = [-1, 0.5]
		RelAbsAssertions.assertArrayRelAbsEquals(new double[] {-1.0, 0.5}, c.constrIneq(x).toColumnArray(), TOL);
		RelAbsAssertions.assertArrayRelAbsEquals(new double[][] { {-1, 0}, {0, -1} }, c.jacIneq(x).toDoubleArray(), TOL);
	}

	@Test
	void testMixedEqAndIneqAndDropped() {
		Matrix A = Matrix.Factory.linkToArray(new double[][] {
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 1},
				{1, 1, 0},
		});
		// row 0: equality   (lb=ub=5)
		// row 1: upper      (lb=-inf, ub=2)
		// row 2: lower      (lb=-3, ub=+inf)
		// row 3: dropped    (lb=-inf, ub=+inf)
		LinearConstraint c = new LinearConstraint(A,
				new double[] {5, NEG_INF, -3, NEG_INF},
				new double[] {5, 2, POS_INF, POS_INF});
		Assertions.assertEquals(1, c.nEq());
		Assertions.assertEquals(2, c.nIneq());
	}

	@Test
	void testTwoSidedSplitsIntoUpperAndLowerRows() {
		// A x = scalar a; constraint 0 <= a <= 2. Internally splits into
		// a - 2 <= 0  (upper)  and  -a + 0 <= 0  (lower).
		Matrix A = Matrix.Factory.linkToArray(new double[][] { {1, 1} });
		LinearConstraint c = new LinearConstraint(A, new double[] {0}, new double[] {2});
		Assertions.assertEquals(0, c.nEq());
		Assertions.assertEquals(2, c.nIneq());

		// At x=(1,1), A x = 2 -- the upper bound is exactly tight.
		Matrix x = Matrix.Factory.linkToArray(new double[] {1.0, 1.0});
		RelAbsAssertions.assertArrayRelAbsEquals(
				new double[] {0.0, -2.0},
				c.constrIneq(x).toColumnArray(),
				TOL);
		// jacIneq rows: +A (upper) and -A (lower).
		RelAbsAssertions.assertArrayRelAbsEquals(
				new double[][] { {1, 1}, {-1, -1} },
				c.jacIneq(x).toDoubleArray(),
				TOL);
	}
}
