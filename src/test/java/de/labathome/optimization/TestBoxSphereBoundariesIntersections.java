/*
 * Copyright 2026 Jonathan Schilling
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package de.labathome.optimization;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.QPSubproblem;
import de.labathome.trustconstr.records.IntersectionResult;


class TestBoxSphereBoundariesIntersections {

	@Test
	void test2dBoxConstraints() {
		final double tolerance = 1.0e-15;

		// Both constraints are active
		IntersectionResult r1 = QPSubproblem.boxSphereIntersections(
				new double[] { 1.0, 1.0},
				new double[] {-2.0, 2.0},
				new double[] {-1.0, -2.0},
				new double[] { 1.0,  2.0}, 2.0);
		RelAbsAssertions.assertRelAbsEquals(0.0, r1.tA(), tolerance);
		RelAbsAssertions.assertRelAbsEquals(0.5, r1.tB(), tolerance);
		Assertions.assertTrue(r1.intersect());

		// None of the constraints are active
		IntersectionResult r2 = QPSubproblem.boxSphereIntersections(
				new double[] { 1.0, 1.0},
				new double[] {-1.0, 1.0},
				new double[] {-1.0, -3.0},
				new double[] { 1.0,  3.0}, 10.0);
		RelAbsAssertions.assertRelAbsEquals(0.0, r2.tA(), tolerance);
		RelAbsAssertions.assertRelAbsEquals(1.0, r2.tB(), tolerance);
		Assertions.assertTrue(r2.intersect());

		// Box constraints are active
		IntersectionResult r3 = QPSubproblem.boxSphereIntersections(
				new double[] { 1.0, 1.0},
				new double[] {-4.0, 4.0},
				new double[] {-1.0, -3.0},
				new double[] { 1.0,  3.0}, 10.0);
		RelAbsAssertions.assertRelAbsEquals(0.0, r3.tA(), tolerance);
		RelAbsAssertions.assertRelAbsEquals(0.5, r3.tB(), tolerance);
		Assertions.assertTrue(r3.intersect());

		// Spherical constraints are active
		IntersectionResult r4 = QPSubproblem.boxSphereIntersections(
				new double[] { 1.0, 1.0},
				new double[] {-4.0, 4.0},
				new double[] {-1.0, -3.0},
				new double[] { 1.0,  3.0}, 2.0);
		RelAbsAssertions.assertRelAbsEquals(0.0, r4.tA(), tolerance);
		RelAbsAssertions.assertRelAbsEquals(0.25, r4.tB(), tolerance);
		Assertions.assertTrue(r4.intersect());

		// Infeasible problems
		IntersectionResult r5a = QPSubproblem.boxSphereIntersections(
				new double[] { 2.0, 2.0},
				new double[] {-4.0, 4.0},
				new double[] {-1.0, -3.0},
				new double[] { 1.0,  3.0}, 2.0);
		Assertions.assertFalse(r5a.intersect());

		IntersectionResult r5b = QPSubproblem.boxSphereIntersections(
				new double[] { 1.0, 1.0},
				new double[] {-4.0, 4.0},
				new double[] {2.0, 4.0},
				new double[] {2.0, 4.0}, 2.0);
		Assertions.assertFalse(r5b.intersect());
	}

	@Test
	void test2dBoxConstraintsEntireLine() {
		final double tolerance = 1.0e-15;

		// Both constraints are active
		IntersectionResult r1 = QPSubproblem.boxSphereIntersections(
				new double[] { 1.0, 1.0},
				new double[] {-2.0, 2.0},
				new double[] {-1.0, -2.0},
				new double[] { 1.0,  2.0}, 2.0, true);
		RelAbsAssertions.assertRelAbsEquals(0.0, r1.tA(), tolerance);
		RelAbsAssertions.assertRelAbsEquals(0.5, r1.tB(), tolerance);
		Assertions.assertTrue(r1.intersect());

		// None of the constraints are active
		IntersectionResult r2 = QPSubproblem.boxSphereIntersections(
				new double[] { 1.0, 1.0},
				new double[] {-1.0, 1.0},
				new double[] {-1.0, -3.0},
				new double[] { 1.0,  3.0}, 10.0, true);
		RelAbsAssertions.assertRelAbsEquals(0.0, r2.tA(), tolerance);
		RelAbsAssertions.assertRelAbsEquals(2.0, r2.tB(), tolerance);
		Assertions.assertTrue(r2.intersect());

		// Box constraints are active
		IntersectionResult r3 = QPSubproblem.boxSphereIntersections(
				new double[] { 1.0, 1.0},
				new double[] {-4.0, 4.0},
				new double[] {-1.0, -3.0},
				new double[] { 1.0,  3.0}, 10.0, true);
		RelAbsAssertions.assertRelAbsEquals(0.0, r3.tA(), tolerance);
		RelAbsAssertions.assertRelAbsEquals(0.5, r3.tB(), tolerance);
		Assertions.assertTrue(r3.intersect());

		// Spherical constraints are active
		IntersectionResult r4 = QPSubproblem.boxSphereIntersections(
				new double[] { 1.0, 1.0},
				new double[] {-4.0, 4.0},
				new double[] {-1.0, -3.0},
				new double[] { 1.0,  3.0}, 2.0, true);
		RelAbsAssertions.assertRelAbsEquals(0.0, r4.tA(), tolerance);
		RelAbsAssertions.assertRelAbsEquals(0.25, r4.tB(), tolerance);
		Assertions.assertTrue(r4.intersect());

		// Infeasible problems
		IntersectionResult r5a = QPSubproblem.boxSphereIntersections(
				new double[] { 2.0, 2.0},
				new double[] {-4.0, 4.0},
				new double[] {-1.0, -3.0},
				new double[] { 1.0,  3.0}, 2.0, true);
		Assertions.assertFalse(r5a.intersect());

		IntersectionResult r5b = QPSubproblem.boxSphereIntersections(
				new double[] { 1.0, 1.0},
				new double[] {-4.0, 4.0},
				new double[] {2.0, 4.0},
				new double[] {2.0, 4.0}, 2.0, true);
		Assertions.assertFalse(r5b.intersect());
	}
}
