package de.labathome.optimization;

import org.junit.jupiter.api.Test;
import org.ujmp.core.Matrix;
import org.ujmp.core.doublematrix.DoubleMatrix2D;

import de.labathome.LinAlg;
import de.labathome.LinearOperator;
import minerva.tests.junit.MinervaAssertions;

class TestModifiedDogleg {

	@Test
	void testCauchyPointEqualToNewtonPoint() {
		final double tolerance = 1.0e-8;

		DoubleMatrix2D A = Matrix.Factory.importFromArray(new double[][] { { 1.0, 8.0 } });
		Matrix b = Matrix.Factory.importFromArray(new double[][] { { -16.0 } });

		LinearOperator[] projections = Projections.projections(A);
		LinearOperator Y = projections[2];

		Matrix newtonPoint = Matrix.Factory.linkToArray(new double[] { 0.24615385, 1.96923077 });

		// Newton point inside boundaries
		Matrix x1 = QuadraticProgrammingSubproblem.modifiedDogleg(A, Y, b, 2.0,
				new double[] {Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY},
				new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY});
		MinervaAssertions.assertArrayRelAbsEquals(LinAlg.col(newtonPoint), LinAlg.col(x1), tolerance);

		// Spherical constraint active
		Matrix x2 = QuadraticProgrammingSubproblem.modifiedDogleg(A, Y, b, 1.0,
				new double[] {Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY},
				new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY});
		Matrix normNewtonPt = newtonPoint.divide(newtonPoint.norm2());
		MinervaAssertions.assertArrayRelAbsEquals(LinAlg.col(normNewtonPt), LinAlg.col(x2), tolerance);

		// Box constraints active
		Matrix x3 = QuadraticProgrammingSubproblem.modifiedDogleg(A, Y, b, 2.0,
				new double[] {Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY},
				new double[] {0.1, Double.POSITIVE_INFINITY});
		Matrix boxPoint = newtonPoint.times(0.1 / newtonPoint.getAsDouble(0, 0));
		MinervaAssertions.assertArrayRelAbsEquals(LinAlg.col(boxPoint), LinAlg.col(x3), tolerance);
	}

	@Test
	void test3dExample() {

	}

}
