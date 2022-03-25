package de.labathome.optimization;

import org.junit.jupiter.api.Test;

import minerva.tests.junit.MinervaAssertions;

class TestEQPDirectFactorization {

	/**
	 * Example 16.2 in Nocedal/Wright, "Numerical Optimization" (2006), p. 452
	 */
	@Test
	void testNocedalExample() {

		final int n = 3;
		final int m = 2;

		final double[][] H = {
				{ 6.0, 2.0, 1.0 },
				{ 2.0, 5.0, 2.0 },
				{ 1.0, 2.0, 4.0 }
		};

		final double[][] A = {
				{ 1.0, 0.0, 1.0 },
				{ 0.0, 1.0, 1.0 }
		};

		final double[] c = { -8.0, -3.0, -3.0 };
		final double[] b = { -3.0, 0.0 }; // This is actually -b.

		EQPProblem eqp = new EQPProblem(n, m, H, c, A, b);
		eqp.directFactorization();

		final double[] x = eqp.getX();
		final double[] lambda = eqp.getLambda();

		final double tolerance = 1.0e-15;
		final double[] expectedX = { 2.0, -1.0, 1.0 };
		final double[] expectedLambda = { 3.0, -2.0 };
		MinervaAssertions.assertArrayRelAbsEquals(expectedX, x, tolerance);
		MinervaAssertions.assertArrayRelAbsEquals(expectedLambda, lambda, tolerance);
	}

}
