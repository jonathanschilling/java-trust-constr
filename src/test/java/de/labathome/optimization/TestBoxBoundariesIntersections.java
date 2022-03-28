package de.labathome.optimization;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import de.labathome.trust_constr.IntersectionResult;
import de.labathome.trust_constr.QuadraticProgrammingSubproblem;
import minerva.tests.junit.MinervaAssertions;

class TestBoxBoundariesIntersections {

	@Test
	void test2dBoxConstraints() {
		final double tolerance = 1.0e-15;

		// Box constraint in the direction of vector d
		IntersectionResult r1 = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {2.0, 0.0},
				new double[] {0.0, 2.0},
				new double[] {1.0, 1.0},
				new double[] {3.0, 3.0});
		MinervaAssertions.assertRelAbsEquals(0.5, r1.tA(), tolerance);
		MinervaAssertions.assertRelAbsEquals(1.0, r1.tB(), tolerance);
		Assertions.assertTrue(r1.intersect());

		// Negative direction
		IntersectionResult r2 = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {2.0, 0.0},
				new double[] {0.0, 2.0},
				new double[] {1.0, -3.0},
				new double[] {3.0, -1.0});
		Assertions.assertFalse(r2.intersect());

		// Some constraints are absent (set to +/- inf)
		IntersectionResult r3 = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {2.0, 0.0},
				new double[] {0.0, 2.0},
				new double[] {Double.NEGATIVE_INFINITY, 1.0},
				new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY});
		MinervaAssertions.assertRelAbsEquals(0.5, r3.tA(), tolerance);
		MinervaAssertions.assertRelAbsEquals(1.0, r3.tB(), tolerance);
		Assertions.assertTrue(r3.intersect());

		// Intersect on the face of the box
		IntersectionResult r4 = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {1.0, 0.0},
				new double[] {0.0, 1.0},
				new double[] {1.0, 1.0},
				new double[] {3.0, 3.0});
		MinervaAssertions.assertRelAbsEquals(1.0, r4.tA(), tolerance);
		MinervaAssertions.assertRelAbsEquals(1.0, r4.tB(), tolerance);
		Assertions.assertTrue(r4.intersect());

		// Interior initial point
		IntersectionResult r5 = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {0.0, 0.0},
				new double[] {4.0, 4.0},
				new double[] {-2.0, -3.0},
				new double[] { 3.0,  2.0});
		MinervaAssertions.assertRelAbsEquals(0.0, r5.tA(), tolerance);
		MinervaAssertions.assertRelAbsEquals(0.5, r5.tB(), tolerance);
		Assertions.assertTrue(r5.intersect());

		// No intersection between line and box constraints
		IntersectionResult r6a = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {2.0, 0.0},
				new double[] {0.0, 2.0},
				new double[] {-3.0, -3.0},
				new double[] {-1.0, -1.0});
		Assertions.assertFalse(r6a.intersect());

		IntersectionResult r6b = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {2.0, 0.0},
				new double[] {0.0, 2.0},
				new double[] {-3.0, 3.0},
				new double[] {-1.0, 1.0});
		Assertions.assertFalse(r6b.intersect());

		IntersectionResult r6c = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {2.0, 0.0},
				new double[] {0.0, 2.0},
				new double[] {-3.0, Double.NEGATIVE_INFINITY},
				new double[] {-1.0, Double.POSITIVE_INFINITY});
		Assertions.assertFalse(r6c.intersect());

		IntersectionResult r6d = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {0.0, 0.0},
				new double[] {1.0, 100.0},
				new double[] {1.0,  1.0},
				new double[] {3.0, 3.0});
		Assertions.assertFalse(r6d.intersect());

		IntersectionResult r6e = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {0.99, 0.0},
				new double[] {0.0,  2.0},
				new double[] {1.0,  1.0},
				new double[] {3.0,  3.0});
		Assertions.assertFalse(r6e.intersect());

		// Initial point on the boundary
		IntersectionResult r7 = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {2.0, 2.0},
				new double[] {0.0, 1.0},
				new double[] {-2.0, -2.0},
				new double[] { 2.0,  2.0});
		MinervaAssertions.assertRelAbsEquals(0.0, r7.tA(), tolerance);
		MinervaAssertions.assertRelAbsEquals(0.0, r7.tB(), tolerance);
		Assertions.assertTrue(r7.intersect());
	}

	@Test
	void test2dBoxConstraintsEntireLine() {
		final double tolerance = 1.0e-15;

		// Box constraint in the direction of vector d
		IntersectionResult r1 = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {2.0, 0.0},
				new double[] {0.0, 2.0},
				new double[] {1.0, 1.0},
				new double[] {3.0, 3.0}, true);
		MinervaAssertions.assertRelAbsEquals(0.5, r1.tA(), tolerance);
		MinervaAssertions.assertRelAbsEquals(1.5, r1.tB(), tolerance);
		Assertions.assertTrue(r1.intersect());

		// Negative direction
		IntersectionResult r2 = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {2.0, 0.0},
				new double[] {0.0, 2.0},
				new double[] {1.0, -3.0},
				new double[] {3.0, -1.0}, true);
		MinervaAssertions.assertRelAbsEquals(-1.5, r2.tA(), tolerance);
		MinervaAssertions.assertRelAbsEquals(-0.5, r2.tB(), tolerance);
		Assertions.assertTrue(r2.intersect());

		// Some constraints are absent (set to +/- inf)
		IntersectionResult r3 = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {2.0, 0.0},
				new double[] {0.0, 2.0},
				new double[] {Double.NEGATIVE_INFINITY, 1.0},
				new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY}, true);
		MinervaAssertions.assertRelAbsEquals(0.5, r3.tA(), tolerance);
		Assertions.assertEquals(Double.POSITIVE_INFINITY, r3.tB());
		Assertions.assertTrue(r3.intersect());

		// Intersect on the face of the box
		IntersectionResult r4 = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {1.0, 0.0},
				new double[] {0.0, 1.0},
				new double[] {1.0, 1.0},
				new double[] {3.0, 3.0}, true);
		MinervaAssertions.assertRelAbsEquals(1.0, r4.tA(), tolerance);
		MinervaAssertions.assertRelAbsEquals(3.0, r4.tB(), tolerance);
		Assertions.assertTrue(r4.intersect());

		// Interior initial pointoint
		IntersectionResult r5 = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {0.0, 0.0},
				new double[] {4.0, 4.0},
				new double[] {-2.0, -3.0},
				new double[] { 3.0,  2.0}, true);
		MinervaAssertions.assertRelAbsEquals(-0.5, r5.tA(), tolerance);
		MinervaAssertions.assertRelAbsEquals( 0.5, r5.tB(), tolerance);
		Assertions.assertTrue(r5.intersect());

        // No intersection between line and box constraints
		IntersectionResult r6a = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {2.0, 0.0},
				new double[] {0.0, 2.0},
				new double[] {-3.0, -3.0},
				new double[] {-1.0, -1.0}, true);
		Assertions.assertFalse(r6a.intersect());

		IntersectionResult r6b = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {2.0, 0.0},
				new double[] {0.0, 2.0},
				new double[] {-3.0, 3.0},
				new double[] {-1.0, 1.0});
		Assertions.assertFalse(r6b.intersect());

		IntersectionResult r6c = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {2.0, 0.0},
				new double[] {0.0, 2.0},
				new double[] {-3.0, Double.NEGATIVE_INFINITY},
				new double[] {-1.0, Double.POSITIVE_INFINITY}, true);
		Assertions.assertFalse(r6c.intersect());

		IntersectionResult r6d = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {0.0, 0.0},
				new double[] {1.0, 100.0},
				new double[] {1.0, 1.0},
				new double[] {3.0, 3.0}, true);
		Assertions.assertFalse(r6d.intersect());

		IntersectionResult r6e = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {0.99, 0.0},
				new double[] {0.0,  2.0},
				new double[] {1.0,  1.0},
				new double[] {3.0,  3.0});
		Assertions.assertFalse(r6e.intersect());

		// Initial point on the boundary
		IntersectionResult r7 = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {2.0, 2.0},
				new double[] {0.0, 1.0},
				new double[] {-2.0, -2.0},
				new double[] { 2.0,  2.0}, true);
		MinervaAssertions.assertRelAbsEquals(-4.0, r7.tA(), tolerance);
		MinervaAssertions.assertRelAbsEquals( 0.0, r7.tB(), tolerance);
		Assertions.assertTrue(r7.intersect());
	}

	@Test
	void test3dBoxConstraints() {
		final double tolerance = 1.0e-15;

		// Simple case
		IntersectionResult r1 = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {1.0, 1.0, 0.0},
				new double[] {0.0, 0.0, 1.0},
				new double[] {1.0, 1.0, 1.0},
				new double[] {3.0, 3.0, 3.0});
		MinervaAssertions.assertRelAbsEquals(1.0, r1.tA(), tolerance);
		MinervaAssertions.assertRelAbsEquals(1.0, r1.tB(), tolerance);
		Assertions.assertTrue(r1.intersect());

		// Negative direction
		IntersectionResult r2 = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {1.0, 1.0,  0.0},
				new double[] {0.0, 0.0, -1.0},
				new double[] {1.0, 1.0, 1.0},
				new double[] {3.0, 3.0, 3.0});
		Assertions.assertFalse(r2.intersect());

		// Interior point
		IntersectionResult r3 = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {2.0,  2.0, 2.0},
				new double[] {0.0, -1.0, 1.0},
				new double[] {1.0, 1.0, 1.0},
				new double[] {3.0, 3.0, 3.0});
		MinervaAssertions.assertRelAbsEquals(0.0, r3.tA(), tolerance);
		MinervaAssertions.assertRelAbsEquals(1.0, r3.tB(), tolerance);
		Assertions.assertTrue(r3.intersect());
	}

	@Test
	void test3dBoxConstraintsEntireLine() {
		final double tolerance = 1.0e-15;

		// Simple case
		IntersectionResult r1 = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {1.0, 1.0, 0.0},
				new double[] {0.0, 0.0, 1.0},
				new double[] {1.0, 1.0, 1.0},
				new double[] {3.0, 3.0, 3.0}, true);
		MinervaAssertions.assertRelAbsEquals(1.0, r1.tA(), tolerance);
		MinervaAssertions.assertRelAbsEquals(3.0, r1.tB(), tolerance);
		Assertions.assertTrue(r1.intersect());

		// Negative direction
		IntersectionResult r2 = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {1.0, 1.0,  0.0},
				new double[] {0.0, 0.0, -1.0},
				new double[] {1.0, 1.0, 1.0},
				new double[] {3.0, 3.0, 3.0}, true);
		MinervaAssertions.assertRelAbsEquals(-3.0, r2.tA(), tolerance);
		MinervaAssertions.assertRelAbsEquals(-1.0, r2.tB(), tolerance);
		Assertions.assertTrue(r2.intersect());

		// Interior point
		IntersectionResult r3 = QuadraticProgrammingSubproblem.boxIntersections(
				new double[] {2.0,  2.0, 2.0},
				new double[] {0.0, -1.0, 1.0},
				new double[] {1.0, 1.0, 1.0},
				new double[] {3.0, 3.0, 3.0}, true);
		MinervaAssertions.assertRelAbsEquals(-1.0, r3.tA(), tolerance);
		MinervaAssertions.assertRelAbsEquals( 1.0, r3.tB(), tolerance);
		Assertions.assertTrue(r3.intersect());
	}
}
