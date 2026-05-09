package de.labathome.optimization;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.QPSubproblem;
import de.labathome.trustconstr.records.IntersectionResult;


class TestSphericalBoundariesIntersections {

	@Test
	void test2dSphereConstraints() {
		final double tolerance = 1.0e-15;

		// Interior inicial point
		IntersectionResult r1 = QPSubproblem.sphereIntersections(
				new double[] {0.0, 0.0},
				new double[] {1.0, 0.0}, 0.5);

		RelAbsAssertions.assertRelAbsEquals(0.0, r1.tA(), tolerance);
		RelAbsAssertions.assertRelAbsEquals(0.5, r1.tB(), tolerance);
		Assertions.assertTrue(r1.intersect());

		// No intersection between line and circle
		IntersectionResult r2 = QPSubproblem.sphereIntersections(
				new double[] {2.0, 0.0},
				new double[] {0.0, 1.0}, 1.0);
		Assertions.assertFalse(r2.intersect());

		// Outside initial point pointing toward outside the circle
		IntersectionResult r3 = QPSubproblem.sphereIntersections(
				new double[] {2.0, 0.0},
				new double[] {1.0, 0.0}, 1.0);
		Assertions.assertFalse(r3.intersect());

		// Outside initial point pointing toward inside the circle
		IntersectionResult r4 = QPSubproblem.sphereIntersections(
				new double[] { 2.0, 0.0},
				new double[] {-1.0, 0.0}, 1.5);

		RelAbsAssertions.assertRelAbsEquals(0.5, r4.tA(), tolerance);
		RelAbsAssertions.assertRelAbsEquals(1.0, r4.tB(), tolerance);
		Assertions.assertTrue(r4.intersect());

		// Initial point on the boundary
		IntersectionResult r5 = QPSubproblem.sphereIntersections(
				new double[] {2.0, 0.0},
				new double[] {1.0, 0.0}, 2.0);

		RelAbsAssertions.assertRelAbsEquals(0.0, r5.tA(), tolerance);
		RelAbsAssertions.assertRelAbsEquals(0.0, r5.tB(), tolerance);
		Assertions.assertTrue(r5.intersect());
	}

	@Test
	void test2dSphereConstraintsLineIntersections() {
		final double tolerance = 1.0e-15;

		// Interior initial point
		IntersectionResult r1 = QPSubproblem.sphereIntersections(
				new double[] {0.0, 0.0},
				new double[] {1.0, 0.0}, 0.5, true);
		RelAbsAssertions.assertRelAbsEquals(-0.5, r1.tA(), tolerance);
		RelAbsAssertions.assertRelAbsEquals( 0.5, r1.tB(), tolerance);
		Assertions.assertTrue(r1.intersect());

		// No intersection between line and circle
		IntersectionResult r2 = QPSubproblem.sphereIntersections(
				new double[] {2.0, 0.0},
				new double[] {0.0, 1.0}, 1.0, false);
		Assertions.assertFalse(r2.intersect());

		// Outside initial point pointing toward outside the circle
		IntersectionResult r3 = QPSubproblem.sphereIntersections(
				new double[] {2.0, 0.0},
				new double[] {1.0, 0.0}, 1.0, true);
		RelAbsAssertions.assertRelAbsEquals(-3.0, r3.tA(), tolerance);
		RelAbsAssertions.assertRelAbsEquals(-1.0, r3.tB(), tolerance);
		Assertions.assertTrue(r3.intersect());

		// Outside initial point pointing toward inside the circle
		IntersectionResult r4 = QPSubproblem.sphereIntersections(
				new double[] { 2.0, 0.0},
				new double[] {-1.0, 0.0}, 1.5, true);
		RelAbsAssertions.assertRelAbsEquals(0.5, r4.tA(), tolerance);
		RelAbsAssertions.assertRelAbsEquals(3.5, r4.tB(), tolerance);
		Assertions.assertTrue(r4.intersect());

		// Initial point on the boundary
		IntersectionResult r5 = QPSubproblem.sphereIntersections(
				new double[] {2.0, 0.0},
				new double[] {1.0, 0.0}, 2.0, true);
		RelAbsAssertions.assertRelAbsEquals(-4.0, r5.tA(), tolerance);
		RelAbsAssertions.assertRelAbsEquals( 0.0, r5.tB(), tolerance);
		Assertions.assertTrue(r5.intersect());
	}
}
