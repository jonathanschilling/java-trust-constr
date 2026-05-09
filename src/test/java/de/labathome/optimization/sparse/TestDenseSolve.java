package de.labathome.optimization.sparse;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.sparse.DenseSolve;

import de.labathome.optimization.RelAbsAssertions;

class TestDenseSolve {

	private static final double TOL = 1.0e-12;

	@Test
	void testSolve3x3() {
		// A = [[2, 1, 1], [1, 3, 2], [1, 0, 0]], b = [4, 5, 6]
		// Solving A x = b gives x = [6, 15, -23]
		double[][] a = {
				{ 2.0, 1.0, 1.0 },
				{ 1.0, 3.0, 2.0 },
				{ 1.0, 0.0, 0.0 },
		};
		double[] b = { 4.0, 5.0, 6.0 };
		double[] x = DenseSolve.solveLU(a, b);
		RelAbsAssertions.assertArrayRelAbsEquals(new double[] {6.0, 15.0, -23.0}, x, TOL);
	}

	@Test
	void testFactorReuse() {
		double[][] a = {
				{ 4.0, 3.0 },
				{ 6.0, 3.0 },
		};
		DenseSolve.LUFactor lu = DenseSolve.factor(a);
		// Solve against the columns of the identity to recover the inverse.
		double[] col0 = lu.solve(new double[] {1.0, 0.0});
		double[] col1 = lu.solve(new double[] {0.0, 1.0});
		// A^{-1} = [[-0.5, 0.5], [1, -2/3]]
		RelAbsAssertions.assertArrayRelAbsEquals(new double[] {-0.5, 1.0}, col0, TOL);
		RelAbsAssertions.assertArrayRelAbsEquals(new double[] {0.5, -2.0 / 3.0}, col1, TOL);
	}

	@Test
	void testSingularMatrixThrows() {
		double[][] a = {
				{ 1.0, 2.0 },
				{ 2.0, 4.0 },
		};
		Assertions.assertThrows(ArithmeticException.class, () -> DenseSolve.factor(a));
	}

	@Test
	void testNonSquareRejected() {
		double[][] a = { {1.0, 2.0, 3.0} };
		Assertions.assertThrows(IllegalArgumentException.class, () -> DenseSolve.factor(a));
	}
}
