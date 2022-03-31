package de.labathome.optimization;

import java.util.Arrays;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.NumDiff;
import org.scipy.optimize.minimize.enums.FiniteDifferenceMethod;
import org.scipy.optimize.minimize.records.AdjustedDifferencingScheme;
import org.ujmp.core.Matrix;

import minerva.tests.junit.MinervaAssertions;

class TestAdjustSchemeToBounds {

	@Test
	void testNoBounds() {
		final double tolerance = 1.0e-15;

		Matrix x0 = Matrix.Factory.zeros(3, 1);
		Matrix h = Matrix.Factory.ones(3, 1).times(1.0e-2);

		Matrix infLower = Matrix.Factory.ones(x0.getSize()).times(Double.NEGATIVE_INFINITY);
		Matrix infUpper = Matrix.Factory.ones(x0.getSize()).times(Double.POSITIVE_INFINITY);

		AdjustedDifferencingScheme ads = NumDiff.adjustSchemeToBounds(x0, h, 1, FiniteDifferenceMethod.ONE_SIDED, infLower, infUpper);
		Matrix hAdjusted = ads.hAdjusted();
		boolean[] useOneSided = ads.useOneSided();

		boolean[] allTrue = new boolean[useOneSided.length];
		Arrays.fill(allTrue, true);
		MinervaAssertions.assertArrayRelAbsEquals(h.toDoubleArray()[0], hAdjusted.toDoubleArray()[0], tolerance);
		Assertions.assertArrayEquals(allTrue, useOneSided);
	}
}
