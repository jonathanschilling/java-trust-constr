package de.labathome.optimization;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.ujmp.core.Matrix;

import de.labathome.LinAlg;
import de.labathome.LinearOperator;
import de.labathome.optimization.PCGResult.PCGStoppingCondition;
import minerva.tests.junit.MinervaAssertions;

class TestProjectedCG {

	/** From Example 16.2 Nocedal/Wright "Numerical Optimization" p.452. */
	@Test
	void testNocedalExample() {
		final double tolerance = 1.0e-15;

		Matrix H = LinAlg.sparse(Matrix.Factory.linkToArray(new double[][] {
			{6, 2, 1},
			{2, 5, 2},
			{1, 2, 4}
		}));

		Matrix A = LinAlg.sparse(Matrix.Factory.linkToArray(new double[][] {
			{1, 0, 1},
			{0, 1, 1}
		}));

		Matrix c = Matrix.Factory.linkToArray(new double[] {-8, -3, -3});

		Matrix b = Matrix.Factory.linkToArray(new double[] {3, 0});

		LinearOperator[] op = Projections.projections(A);
		LinearOperator Z = op[0];
		LinearOperator Y = op[2];

		PCGResult r = QuadraticProgrammingSubproblem.projectedCG(H, c, Z, Y, b);
		Assertions.assertEquals(PCGStoppingCondition.TOLERANCE_SATISFIED, r.stopCond);
		Assertions.assertEquals(false, r.hitsBoundary);
		MinervaAssertions.assertArrayRelAbsEquals(new double[] {2, -1, 1}, LinAlg.col(r.x), tolerance);
	}

	@Test
	void testCompareWithDirectFact() {
		final double tolerance = 1.0e-15;

		Matrix H = LinAlg.sparse(Matrix.Factory.linkToArray(new double[][] {
			{6, 2, 1, 3},
			{2, 5, 2, 4},
			{1, 2, 4, 5},
			{3, 4, 5, 7}
		}));

		Matrix A = LinAlg.sparse(Matrix.Factory.linkToArray(new double[][] {
			{1, 0, 1, 0},
			{0, 1, 1, 1}
		}));

		Matrix c = Matrix.Factory.linkToArray(new double[] {-2, -3, -3, 1});

		Matrix b = Matrix.Factory.linkToArray(new double[] {3, 0});

		LinearOperator[] op = Projections.projections(A);
		LinearOperator Z = op[0];
		LinearOperator Y = op[2];

		double trustRadius = Double.POSITIVE_INFINITY;
		double[] lb = null;
		double[] ub = null;
		double tol = 0.0; // iterate until max iterations
		PCGResult r1 = QuadraticProgrammingSubproblem.projectedCG(H, c, Z, Y, b, trustRadius, lb, ub, tol);
		Matrix[] r2 = QuadraticProgrammingSubproblem.eqpKktFact(H, c, A, b);

		Assertions.assertEquals(PCGStoppingCondition.ITER_LIMIT_REACHED, r1.stopCond);
		Assertions.assertEquals(false, r1.hitsBoundary);
		MinervaAssertions.assertArrayRelAbsEquals(LinAlg.col(r2[0]), LinAlg.col(r1.x), tolerance);
	}

	@Test
	void testTrustRegionInfeasible() {
		Matrix H = LinAlg.sparse(Matrix.Factory.linkToArray(new double[][] {
			{6, 2, 1, 3},
			{2, 5, 2, 4},
			{1, 2, 4, 5},
			{3, 4, 5, 7}
		}));

		Matrix A = LinAlg.sparse(Matrix.Factory.linkToArray(new double[][] {
			{1, 0, 1, 0},
			{0, 1, 1, 1}
		}));

		Matrix c = Matrix.Factory.linkToArray(new double[] {-2, -3, -3, 1});

		Matrix b = Matrix.Factory.linkToArray(new double[] {3, 0});

		LinearOperator[] op = Projections.projections(A);
		LinearOperator Z = op[0];
		LinearOperator Y = op[2];

		double trustRadius = 1.0;
		double[] lb = null;
		double[] ub = null;
		double tol = Double.NaN;

		Assertions.assertThrows(RuntimeException.class, () -> {
			QuadraticProgrammingSubproblem.projectedCG(H, c, Z, Y, b, trustRadius, lb, ub, tol);
		});
	}

	@Test
	void testTrustRegionBarelyFeasible() {
		final double tolerance = 1.0e-15;

		Matrix H = LinAlg.sparse(Matrix.Factory.linkToArray(new double[][] {
			{6, 2, 1, 3},
			{2, 5, 2, 4},
			{1, 2, 4, 5},
			{3, 4, 5, 7}
		}));

		Matrix A = LinAlg.sparse(Matrix.Factory.linkToArray(new double[][] {
			{1, 0, 1, 0},
			{0, 1, 1, 1}
		}));

		Matrix c = Matrix.Factory.linkToArray(new double[] {-2, -3, -3, 1});

		Matrix b = Matrix.Factory.linkToArray(new double[] {3, 0});

		LinearOperator[] op = Projections.projections(A);
		LinearOperator Z = op[0];
		LinearOperator Y = op[2];

		double trustRadius = 2.32379000772445021283;
		double[] lb = null;
		double[] ub = null;
		double tol = 0.0;
		PCGResult r = QuadraticProgrammingSubproblem.projectedCG(H, c, Z, Y, b, trustRadius, lb, ub, tol);

		Assertions.assertEquals(PCGStoppingCondition.TRUST_REGION_BOUNDARY_REACHED, r.stopCond);
		Assertions.assertEquals(true, r.hitsBoundary);
		MinervaAssertions.assertRelAbsEquals(trustRadius, r.x.norm2(), tolerance);
		MinervaAssertions.assertArrayRelAbsEquals(LinAlg.col(Y.apply(b).times(-1)), LinAlg.col(r.x), tolerance);
	}

	@Test
	void testHitsBoundary() {
		final double tolerance = 1.0e-15;

		Matrix H = LinAlg.sparse(Matrix.Factory.linkToArray(new double[][] {
			{6, 2, 1, 3},
			{2, 5, 2, 4},
			{1, 2, 4, 5},
			{3, 4, 5, 7}
		}));

		Matrix A = LinAlg.sparse(Matrix.Factory.linkToArray(new double[][] {
			{1, 0, 1, 0},
			{0, 1, 1, 1}
		}));

		Matrix c = Matrix.Factory.linkToArray(new double[] {-2, -3, -3, 1});

		Matrix b = Matrix.Factory.linkToArray(new double[] {3, 0});

		LinearOperator[] op = Projections.projections(A);
		LinearOperator Z = op[0];
		LinearOperator Y = op[2];

		double trustRadius = 3.0;
		double[] lb = null;
		double[] ub = null;
		double tol = 0.0;
		PCGResult r = QuadraticProgrammingSubproblem.projectedCG(H, c, Z, Y, b, trustRadius, lb, ub, tol);
		Assertions.assertEquals(PCGStoppingCondition.TRUST_REGION_BOUNDARY_REACHED, r.stopCond);
		Assertions.assertEquals(true, r.hitsBoundary);
		MinervaAssertions.assertRelAbsEquals(trustRadius, r.x.norm2(), tolerance);
	}

	@Test
	void testNegativeCurvatureUnconstrained() {
		Matrix H = LinAlg.sparse(Matrix.Factory.linkToArray(new double[][] {
			{1, 2, 1, 3},
			{2, 0, 2, 4},
			{1, 2, 0, 2},
			{3, 4, 2, 0}
		}));

		Matrix A = LinAlg.sparse(Matrix.Factory.linkToArray(new double[][] {
			{1, 0, 1, 0},
			{0, 1, 0, 1}
		}));

		Matrix c = Matrix.Factory.linkToArray(new double[] {-2, -3, -3, 1});

		Matrix b = Matrix.Factory.linkToArray(new double[] {3, 0});

		LinearOperator[] op = Projections.projections(A);
		LinearOperator Z = op[0];
		LinearOperator Y = op[2];

		double trustRadius = Double.POSITIVE_INFINITY;
		double[] lb = null;
		double[] ub = null;
		double tol = 0.0;
		Assertions.assertThrows(RuntimeException.class, () -> {
			QuadraticProgrammingSubproblem.projectedCG(H, c, Z, Y, b, trustRadius, lb, ub, tol);
		});
	}

	@Test
	void testNegativeCurvature() {
		final double tolerance = 1.0e-15;

		Matrix H = LinAlg.sparse(Matrix.Factory.linkToArray(new double[][] {
			{1, 2, 1, 3},
			{2, 0, 2, 4},
			{1, 2, 0, 2},
			{3, 4, 2, 0}
		}));

		Matrix A = LinAlg.sparse(Matrix.Factory.linkToArray(new double[][] {
			{1, 0, 1, 0},
			{0, 1, 0, 1}
		}));

		Matrix c = Matrix.Factory.linkToArray(new double[] {-2, -3, -3, 1});

		Matrix b = Matrix.Factory.linkToArray(new double[] {3, 0});

		LinearOperator[] op = Projections.projections(A);
		LinearOperator Z = op[0];
		LinearOperator Y = op[2];

		double trustRadius = 1000.0;
		double[] lb = null;
		double[] ub = null;
		double tol = 0.0;
		PCGResult r = QuadraticProgrammingSubproblem.projectedCG(H, c, Z, Y, b, trustRadius, lb, ub, tol);
		Assertions.assertEquals(PCGStoppingCondition.NEGATIVE_CURVATURE, r.stopCond);
		Assertions.assertEquals(true, r.hitsBoundary);
		MinervaAssertions.assertRelAbsEquals(trustRadius, r.x.norm2(), tolerance);
	}

	@Test
	void testInactiveBoxConstraints() {

	}

	@Test
	void testActiveBoxConstraintsMaximumIterationsReached() {

	}

	@Test
	void testActiveBoxConstraintsHitsBoundaries() {

	}

	@Test
	void testActiveBoxConstraintsHitsBoundariesInfeasibleIter() {

	}

	@Test
	void testActiveBoxConstraintsNegativeCurvature() {

	}

}
