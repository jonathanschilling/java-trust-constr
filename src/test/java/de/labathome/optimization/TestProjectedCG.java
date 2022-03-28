package de.labathome.optimization;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.ujmp.core.Matrix;

import de.labathome.trust_constr.CGInfo;
import de.labathome.trust_constr.LinAlg;
import de.labathome.trust_constr.LinearOperator;
import de.labathome.trust_constr.PCGStoppingCondition;
import de.labathome.trust_constr.Projections;
import de.labathome.trust_constr.QPSubproblem;
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

		Matrix b = Matrix.Factory.linkToArray(new double[] {3, 0}).times(-1); // TODO: fix -1 in b !!!

		LinearOperator[] op = Projections.projections(A);
		LinearOperator Z = op[0];
		LinearOperator Y = op[2];

		CGInfo r = QPSubproblem.projectedCG(LinAlg.op(H), c, Z, Y, b);
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

		Matrix b = Matrix.Factory.linkToArray(new double[] {3, 0}).times(-1); // TODO: fix -1 in b !!!

		LinearOperator[] op = Projections.projections(A);
		LinearOperator Z = op[0];
		LinearOperator Y = op[2];

		double trustRadius = Double.POSITIVE_INFINITY;
		double[] lb = null;
		double[] ub = null;
		double tol = 0.0; // iterate until max iterations
		CGInfo r1 = QPSubproblem.projectedCG(LinAlg.op(H), c, Z, Y, b, trustRadius, lb, ub, tol);
		Assertions.assertEquals(PCGStoppingCondition.ITER_LIMIT_REACHED, r1.stopCond);
		Assertions.assertEquals(false, r1.hitsBoundary);

		Matrix[] r2 = QPSubproblem.eqpKktFact(H, c, A, b);
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

		Matrix b = Matrix.Factory.linkToArray(new double[] {3, 0}).times(-1); // TODO: fix -1 in b !!!

		LinearOperator[] op = Projections.projections(A);
		LinearOperator Z = op[0];
		LinearOperator Y = op[2];

		double trustRadius = 1.0;
		double[] lb = null;
		double[] ub = null;
		double tol = Double.NaN;

		Assertions.assertThrows(RuntimeException.class, () -> {
			QPSubproblem.projectedCG(LinAlg.op(H), c, Z, Y, b, trustRadius, lb, ub, tol);
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

		Matrix b = Matrix.Factory.linkToArray(new double[] {3, 0}).times(-1); // TODO: fix -1 in b !!!

		LinearOperator[] op = Projections.projections(A);
		LinearOperator Z = op[0];
		LinearOperator Y = op[2];

		double trustRadius = 2.32379000772445021283;
		double[] lb = null;
		double[] ub = null;
		double tol = 0.0;
		CGInfo r = QPSubproblem.projectedCG(LinAlg.op(H), c, Z, Y, b, trustRadius, lb, ub, tol);

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

		Matrix b = Matrix.Factory.linkToArray(new double[] {3, 0}).times(-1); // TODO: fix -1 in b !!!

		LinearOperator[] op = Projections.projections(A);
		LinearOperator Z = op[0];
		LinearOperator Y = op[2];

		double trustRadius = 3.0;
		double[] lb = null;
		double[] ub = null;
		double tol = 0.0;
		CGInfo r = QPSubproblem.projectedCG(LinAlg.op(H), c, Z, Y, b, trustRadius, lb, ub, tol);
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

		Matrix b = Matrix.Factory.linkToArray(new double[] {3, 0}).times(-1); // TODO: fix -1 in b !!!

		LinearOperator[] op = Projections.projections(A);
		LinearOperator Z = op[0];
		LinearOperator Y = op[2];

		double trustRadius = Double.POSITIVE_INFINITY;
		double[] lb = null;
		double[] ub = null;
		double tol = 0.0;
		Assertions.assertThrows(RuntimeException.class, () -> {
			QPSubproblem.projectedCG(LinAlg.op(H), c, Z, Y, b, trustRadius, lb, ub, tol);
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

		Matrix b = Matrix.Factory.linkToArray(new double[] {3, 0}).times(-1); // TODO: fix -1 in b !!!

		LinearOperator[] op = Projections.projections(A);
		LinearOperator Z = op[0];
		LinearOperator Y = op[2];

		double trustRadius = 1000.0;
		double[] lb = null;
		double[] ub = null;
		double tol = 0.0;
		CGInfo r = QPSubproblem.projectedCG(LinAlg.op(H), c, Z, Y, b, trustRadius, lb, ub, tol);
		Assertions.assertEquals(PCGStoppingCondition.NEGATIVE_CURVATURE, r.stopCond);
		Assertions.assertEquals(true, r.hitsBoundary);
		MinervaAssertions.assertRelAbsEquals(trustRadius, r.x.norm2(), tolerance);
	}

	/**
	 * The box constraints are inactive at the solution
	 * but are active during the iterations.
	 */
	@Test
	void testInactiveBoxConstraints() {
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

		Matrix b = Matrix.Factory.linkToArray(new double[] {3, 0}).times(-1); // TODO: fix -1 in b !!!

		LinearOperator[] op = Projections.projections(A);
		LinearOperator Z = op[0];
		LinearOperator Y = op[2];

		double trustRadius = Double.POSITIVE_INFINITY;
		double[] lb = new double[] {0.5, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY};
		double[] ub = null;
		double tol = 0.0;
		CGInfo r = QPSubproblem.projectedCG(LinAlg.op(H), c, Z, Y, b, trustRadius, lb, ub, tol);
		Assertions.assertEquals(PCGStoppingCondition.ITER_LIMIT_REACHED, r.stopCond);
		Assertions.assertEquals(false, r.hitsBoundary);

		Matrix[] r2 = QPSubproblem.eqpKktFact(H, c, A, b);
		MinervaAssertions.assertArrayRelAbsEquals(LinAlg.col(r2[0]), LinAlg.col(r.x), tolerance);
	}

	/**
	 * The box constraints active and the termination is
     * by maximum iterations (infeasible iteraction).
	 */
	@Test
	void testActiveBoxConstraintsMaximumIterationsReached() {
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

		Matrix b = Matrix.Factory.linkToArray(new double[] {3, 0}).times(-1); // TODO: fix -1 in b !!!

		LinearOperator[] op = Projections.projections(A);
		LinearOperator Z = op[0];
		LinearOperator Y = op[2];

		double trustRadius = Double.POSITIVE_INFINITY;
		double[] lb = new double[] {0.8, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY};
		double[] ub = null;
		double tol = 0.0;
		CGInfo r = QPSubproblem.projectedCG(LinAlg.op(H), c, Z, Y, b, trustRadius, lb, ub, tol);
		Assertions.assertEquals(PCGStoppingCondition.ITER_LIMIT_REACHED, r.stopCond);
		Assertions.assertEquals(true, r.hitsBoundary);

		MinervaAssertions.assertArrayRelAbsEquals(LinAlg.col(b.times(-1)), LinAlg.col(A.mtimes(r.x.transpose())), tolerance);
		MinervaAssertions.assertRelAbsEquals(0.8, r.x.getAsDouble(0, 0), tolerance);
	}

	/**
	 * The box constraints are active and the termination is
     * because it hits boundary (without infeasible iteraction).
	 */
	@Test
	void testActiveBoxConstraintsHitsBoundaries() {
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

		Matrix b = Matrix.Factory.linkToArray(new double[] {3, 0}).times(-1); // TODO: fix -1 in b !!!

		LinearOperator[] op = Projections.projections(A);
		LinearOperator Z = op[0];
		LinearOperator Y = op[2];

		double trustRadius = 3.0;
		double[] lb = null;
		double[] ub = new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, 1.6, Double.POSITIVE_INFINITY};
		double tol = 0.0;
		CGInfo r = QPSubproblem.projectedCG(LinAlg.op(H), c, Z, Y, b, trustRadius, lb, ub, tol);
		Assertions.assertEquals(PCGStoppingCondition.TRUST_REGION_BOUNDARY_REACHED, r.stopCond);
		Assertions.assertEquals(true, r.hitsBoundary);

		MinervaAssertions.assertRelAbsEquals(1.6, r.x.getAsDouble(0, 2), tolerance);
	}

	/**
	 * The box constraints are active and the termination is
     * because it hits boundary (infeasible iteraction).
	 */
	@Test
	void testActiveBoxConstraintsHitsBoundariesInfeasibleIter() {
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

		Matrix b = Matrix.Factory.linkToArray(new double[] {3, 0}).times(-1); // TODO: fix -1 in b !!!

		LinearOperator[] op = Projections.projections(A);
		LinearOperator Z = op[0];
		LinearOperator Y = op[2];

		double trustRadius = 4.0;
		double[] lb = null;
		double[] ub = new double[] {Double.POSITIVE_INFINITY, 0.1, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY};
		double tol = 0.0;
		CGInfo r = QPSubproblem.projectedCG(LinAlg.op(H), c, Z, Y, b, trustRadius, lb, ub, tol);
		Assertions.assertEquals(PCGStoppingCondition.TRUST_REGION_BOUNDARY_REACHED, r.stopCond);
		Assertions.assertEquals(true, r.hitsBoundary);

		MinervaAssertions.assertRelAbsEquals(0.1, r.x.getAsDouble(0, 1), tolerance);
	}

	/**
	 * The box constraints are active and the termination is
     * because it hits boundary (no infeasible iteraction).
	 */
	@Test
	void testActiveBoxConstraintsNegativeCurvature() {
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

		Matrix b = Matrix.Factory.linkToArray(new double[] {3, 0}).times(-1); // TODO: fix -1 in b !!!

		LinearOperator[] op = Projections.projections(A);
		LinearOperator Z = op[0];
		LinearOperator Y = op[2];

		double trustRadius = 1000.0;
		double[] lb = null;
		double[] ub = new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, 100.0, Double.POSITIVE_INFINITY};
		double tol = 0.0;
		CGInfo r = QPSubproblem.projectedCG(LinAlg.op(H), c, Z, Y, b, trustRadius, lb, ub, tol);
		Assertions.assertEquals(PCGStoppingCondition.NEGATIVE_CURVATURE, r.stopCond);
		Assertions.assertEquals(true, r.hitsBoundary);
		MinervaAssertions.assertRelAbsEquals(100.0, r.x.getAsDouble(0, 2), tolerance);
	}
}
