package de.labathome.optimization;

import org.junit.jupiter.api.Test;
import org.ujmp.core.Matrix;
import org.ujmp.core.doublematrix.DoubleMatrix2D;

import minerva.tests.junit.MinervaAssertions;

class TestModifiedDogleg {

	@Test
	void testCauchyPointEqualToNewtonPoint() {
		final double tolerance = 1.0e-15;

		DoubleMatrix2D A = Matrix.Factory.importFromArray(new double[][] { { 1.0, 8.0 } });
		Matrix b = Matrix.Factory.importFromArray(new double[][] { { -16.0 } });

		Matrix[] projections = Projections.projections(A);
		Matrix Y = projections[2];

		double[] newtonPoint = { 0.24615385, 1.96923077 };

		// Newton point inside boundaries
		Matrix x1 = QuadraticProgrammingSubproblem.modifiedDogleg(A, Y, b, 2.0,
				new double[] {Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY},
				new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY});

		double[] x1Arr = x1.transpose().toDoubleArray()[0];
		MinervaAssertions.assertArrayRelAbsEquals(newtonPoint, x1Arr, tolerance);

		// TODO: Spherical constraint active

		// TODO: Box constraints active
	}

}
