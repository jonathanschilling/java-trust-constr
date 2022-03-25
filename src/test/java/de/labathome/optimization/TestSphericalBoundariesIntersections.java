package de.labathome.optimization;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import minerva.tests.junit.MinervaAssertions;

class TestSphericalBoundariesIntersections {

	@Test
	void test2dSphereConstraints() {

		// Interior inicial point
		IntersectionResult r1 = EQPProblem.sphereIntersections(
				new double[] {0.0, 0.0},
				new double[] {1.0, 0.0}, 0.5);

		final double tolerance = 1.0e-15;
		MinervaAssertions.assertRelAbsEquals(0.0, r1.tA(), tolerance);
		MinervaAssertions.assertRelAbsEquals(0.5, r1.tB(), tolerance);
		Assertions.assertTrue(r1.intersect());


	}
}
