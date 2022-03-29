package de.labathome.optimization;

import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.LinAlg;
import org.scipy.optimize.minimize.Projections;
import org.scipy.optimize.minimize.QPSubproblem;
import org.scipy.optimize.minimize.interfaces.LinearOperator;
import org.ujmp.core.Matrix;
import org.ujmp.core.doublematrix.DoubleMatrix2D;

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
		Matrix x1 = QPSubproblem.modifiedDogleg(A, Y, b, 2.0,
				new double[] {Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY},
				new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY});
		MinervaAssertions.assertArrayRelAbsEquals(LinAlg.col(newtonPoint), LinAlg.col(x1), tolerance);

		// Spherical constraint active
		Matrix x2 = QPSubproblem.modifiedDogleg(A, Y, b, 1.0,
				new double[] {Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY},
				new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY});
		Matrix normNewtonPt = newtonPoint.divide(newtonPoint.norm2());
		MinervaAssertions.assertArrayRelAbsEquals(LinAlg.col(normNewtonPt), LinAlg.col(x2), tolerance);

		// Box constraints active
		Matrix x3 = QPSubproblem.modifiedDogleg(A, Y, b, 2.0,
				new double[] {Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY},
				new double[] {0.1, Double.POSITIVE_INFINITY});
		Matrix boxPoint = newtonPoint.times(0.1 / newtonPoint.getAsDouble(0, 0));
		MinervaAssertions.assertArrayRelAbsEquals(LinAlg.col(boxPoint), LinAlg.col(x3), tolerance);
	}

	@Test
	void test3dExample() {
		// TODO: bad tolerance only because constants below have too few digits...
		final double tolerance = 1.0e-7;

		Matrix A = Matrix.Factory.linkToArray(new double[][] {
			{1, 8, 1},
			{4, 2, 2}
		});

		Matrix b = Matrix.Factory.linkToArray(new double[] {-16, 2});

		LinearOperator[] op = Projections.projections(A);
		LinearOperator Y = op[2];

		Matrix newtonPoint = Matrix.Factory.linkToArray(new double[] {-1.37090909, 2.23272727, -0.49090909});
		Matrix cauchyPoint = Matrix.Factory.linkToArray(new double[] { 0.11165723, 1.73068711,  0.16748585});
		Matrix origin = Matrix.Factory.zeros(newtonPoint.getRowCount(), newtonPoint.getColumnCount());

		// newton_point inside boundaries
		Matrix x1 = QPSubproblem.modifiedDogleg(A, Y, b, 3.0,
				new double[] {Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY},
				new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY});
		MinervaAssertions.assertArrayRelAbsEquals(LinAlg.col(newtonPoint), LinAlg.col(x1), tolerance);

		// line between cauchy_point and newton_point contains best point (spherical constraint is active).
		Matrix x2 = QPSubproblem.modifiedDogleg(A, Y, b, 2.0,
				new double[] {Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY},
				new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY});
		Matrix z = cauchyPoint;
		Matrix d = newtonPoint.minus(cauchyPoint);
		Matrix t = x2.minus(z).divide(d);
		MinervaAssertions.assertArrayRelAbsEquals(new double[] {0.40807330, 0.40807330, 0.40807330}, LinAlg.col(t), tolerance);
		MinervaAssertions.assertRelAbsEquals(2.0, x2.norm2(), tolerance);

		// line between cauchy_point and newton_point contains best point (box constraint is active).
		Matrix x3 = QPSubproblem.modifiedDogleg(A, Y, b, 5.0,
				new double[] {-1.0, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY},
				new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY});
		z = cauchyPoint;
		d = newtonPoint.minus(cauchyPoint);
		t = x3.minus(z).divide(d);
		MinervaAssertions.assertArrayRelAbsEquals(new double[] {0.7498195, 0.7498195, 0.7498195}, LinAlg.col(t), tolerance);
		MinervaAssertions.assertRelAbsEquals(-1.0, x3.getAsDouble(0, 0), tolerance);

		// line between origin and cauchy_point contains best point (spherical constraint is active).
		Matrix x4 = QPSubproblem.modifiedDogleg(A, Y, b, 1.0,
				new double[] {Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY},
				new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY});
		z = origin;
		d = cauchyPoint;
		t = x4.minus(z).divide(d);
		MinervaAssertions.assertArrayRelAbsEquals(new double[] {0.573936265, 0.573936265, 0.573936265}, LinAlg.col(t), tolerance);
		MinervaAssertions.assertRelAbsEquals(1.0, x4.norm2(), tolerance);

		// line between origin and newton_point contains best point (box constraint is active).
		Matrix x5 = QPSubproblem.modifiedDogleg(A, Y, b, 2.0,
				new double[] {Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY},
				new double[] {Double.POSITIVE_INFINITY, 1.0, Double.POSITIVE_INFINITY});
		z = origin;
		d = newtonPoint;
		t = x5.minus(z).divide(d);
		MinervaAssertions.assertArrayRelAbsEquals(new double[] {0.4478827364, 0.4478827364, 0.4478827364}, LinAlg.col(t), tolerance);
		MinervaAssertions.assertRelAbsEquals(1.0, x5.getAsDouble(1, 0), tolerance);
	}
}
