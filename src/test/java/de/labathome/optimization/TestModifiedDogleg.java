package de.labathome.optimization;

import org.junit.jupiter.api.Test;

import de.labathome.LinearOperator;
import minerva.tests.junit.MinervaAssertions;

class TestModifiedDogleg {

	@Test
	void testCauchyPointEqualToNewtonPoint() {
		final double tolerance = 1.0e-15;

		final double[][] A = { { 1.0, 8.0 } };
		final double[] b = { -16.0 };

		LinearOperator[] projections = Projections.projections(A);
		LinearOperator Y = projections[2];

		double[] newtonPoint = { 0.24615385, 1.96923077 };

		// Newton point inside boundaries
		double[] x1 = QuadraticProgrammingSubproblem.modifiedDogleg(A, Y.mat(), b, 2.0,
				new double[] {Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY},
				new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY});
		MinervaAssertions.assertArrayRelAbsEquals(newtonPoint, x1, tolerance);

		// TODO: Spherical constraint active

		// TODO: Box constraints active
	}

}
