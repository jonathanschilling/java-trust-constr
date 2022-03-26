package de.labathome.optimization;

import org.junit.jupiter.api.Test;
import org.ujmp.core.Matrix;

import minerva.tests.junit.MinervaAssertions;

class TestEQPDirectFactorization {

	/**
	 * Example 16.2 in Nocedal/Wright, "Numerical Optimization" (2006), p. 452
	 */
	@Test
	void testNocedalExample() {
		final double tolerance = 1.0e-15;

		final int n = 3;
		final int m = 2;

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

		QuadraticProgrammingSubproblem eqp = new QuadraticProgrammingSubproblem(n, m, H, c, A, b);
		eqp.directFactorization();

		final double[] x = eqp.getX().transpose().toDoubleArray()[0];
		final double[] lambda = eqp.getLambda().transpose().toDoubleArray()[0];

		final double[] expectedX = { 2.0, -1.0, 1.0 };
		final double[] expectedLambda = { 3.0, -2.0 };
		MinervaAssertions.assertArrayRelAbsEquals(expectedX, x, tolerance);
		MinervaAssertions.assertArrayRelAbsEquals(expectedLambda, lambda, tolerance);
	}

}
