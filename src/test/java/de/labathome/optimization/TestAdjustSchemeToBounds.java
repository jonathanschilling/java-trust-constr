package de.labathome.optimization;

import java.util.Arrays;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.LinAlg;
import org.scipy.optimize.minimize.NumDiff;
import org.scipy.optimize.minimize.enums.FiniteDifferenceMethod;
import org.scipy.optimize.minimize.records.AdjustedDifferencingScheme;
import org.ujmp.core.Matrix;

import minerva.tests.junit.MinervaAssertions;

class TestAdjustSchemeToBounds {

	@Test
	void testNoBounds() {
		final double tolerance = 1.0e-15;

		int n = 3;
		Matrix x0 = Matrix.Factory.zeros(n, 1);
		Matrix h = Matrix.Factory.ones(n, 1).times(1.0e-2);

		Matrix infLower = Matrix.Factory.ones(x0.getSize()).times(Double.NEGATIVE_INFINITY);
		Matrix infUpper = Matrix.Factory.ones(x0.getSize()).times(Double.POSITIVE_INFINITY);

		boolean[] allFalse = new boolean[n];
		boolean[] allTrue = new boolean[n];
		Arrays.fill(allTrue, true);

		// 1. one-sided, 1 step
		AdjustedDifferencingScheme ads = NumDiff.adjustSchemeToBounds(x0, h, 1, FiniteDifferenceMethod.ONE_SIDED, infLower, infUpper);
		MinervaAssertions.assertArrayRelAbsEquals(LinAlg.col(h), LinAlg.col(ads.hAdjusted()), tolerance);
		Assertions.assertArrayEquals(allTrue, ads.useOneSided());

		// 2. one-sided, 2 steps
		ads = NumDiff.adjustSchemeToBounds(x0, h, 2, FiniteDifferenceMethod.ONE_SIDED, infLower, infUpper);
		MinervaAssertions.assertArrayRelAbsEquals(LinAlg.col(h), LinAlg.col(ads.hAdjusted()), tolerance);
		Assertions.assertArrayEquals(allTrue, ads.useOneSided());

		// 3. two-sided, 1 step
		ads = NumDiff.adjustSchemeToBounds(x0, h, 1, FiniteDifferenceMethod.TWO_SIDED, infLower, infUpper);
		MinervaAssertions.assertArrayRelAbsEquals(LinAlg.col(h), LinAlg.col(ads.hAdjusted()), tolerance);
		Assertions.assertArrayEquals(allFalse, ads.useOneSided());

		// 4. two-sided, 2 steps
		ads = NumDiff.adjustSchemeToBounds(x0, h, 2, FiniteDifferenceMethod.TWO_SIDED, infLower, infUpper);
		MinervaAssertions.assertArrayRelAbsEquals(LinAlg.col(h), LinAlg.col(ads.hAdjusted()), tolerance);
		Assertions.assertArrayEquals(allFalse, ads.useOneSided());
	}

	@Test
	void testWithBound() {
		final double tolerance = 1.0e-15;

		int n = 3;
		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.0, 0.85, -0.85});
		Matrix lb = Matrix.Factory.ones(n, 1).times(-1);
		Matrix ub = Matrix.Factory.ones(n, 1);
		Matrix h = Matrix.Factory.linkToArray(new double[] {1, 1, -1}).times(1.0e-1);

		boolean[] allFalse = new boolean[n];

		AdjustedDifferencingScheme ads = NumDiff.adjustSchemeToBounds(x0, h, 1, FiniteDifferenceMethod.ONE_SIDED, lb, ub);
		MinervaAssertions.assertArrayRelAbsEquals(LinAlg.col(h), LinAlg.col(ads.hAdjusted()), tolerance);

		ads = NumDiff.adjustSchemeToBounds(x0, h, 2, FiniteDifferenceMethod.ONE_SIDED, lb, ub);
		MinervaAssertions.assertArrayRelAbsEquals(new double[] {0.1, -0.1, 0.1}, LinAlg.col(ads.hAdjusted()), tolerance);

		ads = NumDiff.adjustSchemeToBounds(x0, h, 1, FiniteDifferenceMethod.TWO_SIDED, lb, ub);
		MinervaAssertions.assertArrayRelAbsEquals(LinAlg.col(h), LinAlg.col(ads.hAdjusted()), tolerance);
		Assertions.assertArrayEquals(allFalse, ads.useOneSided());

		ads = NumDiff.adjustSchemeToBounds(x0, h, 2, FiniteDifferenceMethod.TWO_SIDED, lb, ub);
		MinervaAssertions.assertArrayRelAbsEquals(new double[] {0.1, -0.1, 0.1}, LinAlg.col(ads.hAdjusted()), tolerance);
		Assertions.assertArrayEquals(new boolean[] {false, true, true}, ads.useOneSided());
	}
}
