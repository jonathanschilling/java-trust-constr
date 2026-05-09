package de.labathome.optimization.integration;

import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.LinearConstraint;
import org.scipy.optimize.minimize.MinimizeTrustConstr;
import org.scipy.optimize.minimize.records.OptimizeResult;
import org.ujmp.core.Matrix;

import de.labathome.optimization.RelAbsAssertions;

/**
 * End-to-end smoke tests for {@link MinimizeTrustConstr#minimizeEqualityConstrained}.
 *
 * <p>Each test exercises the SQP loop, the projection KKT path
 * ({@code Projections.augmentedSystemProjections} → CSR + dense LAPACK), and
 * the Lagrangian Hessian → modified-dogleg → projected-CG → step-acceptance
 * pipeline.
 *
 * <p>Both problems converge to machine precision; iteration counts match
 * scipy's reference (3 for the quadratic, 7 for Rosenbrock).
 */
class TestEqualityConstrainedRosenbrock {

	private static final double TIGHT_TOL = 1.0e-12;

	private static double rosenbrock(Matrix x) {
		double a = x.getAsDouble(0, 0);
		double b = x.getAsDouble(1, 0);
		return (1.0 - a) * (1.0 - a) + 100.0 * (b - a * a) * (b - a * a);
	}

	private static Matrix rosenbrockGrad(Matrix x) {
		double a = x.getAsDouble(0, 0);
		double b = x.getAsDouble(1, 0);
		double[] g = new double[2];
		g[0] = -2.0 * (1.0 - a) - 400.0 * a * (b - a * a);
		g[1] = 200.0 * (b - a * a);
		return Matrix.Factory.linkToArray(g);
	}

	private static Matrix rosenbrockHess(Matrix x) {
		double a = x.getAsDouble(0, 0);
		double b = x.getAsDouble(1, 0);
		double[][] H = new double[2][2];
		H[0][0] = 2.0 - 400.0 * b + 1200.0 * a * a;
		H[0][1] = -400.0 * a;
		H[1][0] = -400.0 * a;
		H[1][1] = 200.0;
		return Matrix.Factory.linkToArray(H);
	}

	@Test
	void simpleQuadraticOnHyperplane() {
		// minimize x^2 + y^2 subject to x + y = 2  ->  optimum at (1, 1), f = 2
		java.util.function.Function<Matrix, Double> q = x ->
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0) + x.getAsDouble(1, 0) * x.getAsDouble(1, 0);
		java.util.function.Function<Matrix, Matrix> qg = x ->
				Matrix.Factory.linkToArray(new double[] { 2 * x.getAsDouble(0, 0), 2 * x.getAsDouble(1, 0) });
		java.util.function.Function<Matrix, Matrix> qh = x ->
				Matrix.Factory.linkToArray(new double[][] { {2, 0}, {0, 2} });

		Matrix A = Matrix.Factory.linkToArray(new double[][] { { 1.0, 1.0 } });
		LinearConstraint eq = new LinearConstraint(A, new double[] {2.0}, new double[] {2.0});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {-1.0, 0.5});
		OptimizeResult r = MinimizeTrustConstr.minimizeEqualityConstrained(
				q, qg, qh, x0, eq, 100, 1.0e-10, 1.0e-10);

		RelAbsAssertions.assertArrayRelAbsEquals(new double[] {1.0, 1.0},
				new double[] {r.x.getAsDouble(0, 0), r.x.getAsDouble(1, 0)}, TIGHT_TOL);
		RelAbsAssertions.assertRelAbsEquals(2.0, r.fun, TIGHT_TOL);
		// scipy converges in 3 iterations.
		org.junit.jupiter.api.Assertions.assertEquals(3, r.nIter);
		// Lagrangian gradient at the optimum should be tiny.
		org.junit.jupiter.api.Assertions.assertNotNull(r.lagrangianGrad);
		org.junit.jupiter.api.Assertions.assertTrue(r.lagrangianGrad.normInf() < 1e-10,
				"Lagrangian gradient too large: " + r.lagrangianGrad.normInf());
		// Eval counters should be modest and nonzero.
		org.junit.jupiter.api.Assertions.assertTrue(r.numFunctionEval > 0 && r.numFunctionEval < 50,
				"Unexpected fun-eval count: " + r.numFunctionEval);
		org.junit.jupiter.api.Assertions.assertTrue(r.numJacobianEval > 0 && r.numJacobianEval < 50,
				"Unexpected grad-eval count: " + r.numJacobianEval);
		org.junit.jupiter.api.Assertions.assertTrue(r.numHessianEval > 0 && r.numHessianEval < 50,
				"Unexpected hess-eval count: " + r.numHessianEval);
	}

	@Test
	void rosenbrockOnHyperplane() {
		// Constraint: x + y == 2.  The unconstrained Rosenbrock minimum (1, 1)
		// already lies on this hyperplane, so the trust-constr solver should find it.
		Matrix A = Matrix.Factory.linkToArray(new double[][] { { 1.0, 1.0 } });
		LinearConstraint eq = new LinearConstraint(A, new double[] {2.0}, new double[] {2.0});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.5});
		OptimizeResult r = MinimizeTrustConstr.minimizeEqualityConstrained(
				TestEqualityConstrainedRosenbrock::rosenbrock,
				TestEqualityConstrainedRosenbrock::rosenbrockGrad,
				TestEqualityConstrainedRosenbrock::rosenbrockHess,
				x0, eq,
				1000, 1.0e-10, 1.0e-10);

		RelAbsAssertions.assertArrayRelAbsEquals(
				new double[] {1.0, 1.0},
				new double[] {r.x.getAsDouble(0, 0), r.x.getAsDouble(1, 0)},
				TIGHT_TOL);
		RelAbsAssertions.assertRelAbsEquals(0.0, r.fun, TIGHT_TOL);
		// scipy converges in 7 iterations from this start; we should match.
		org.junit.jupiter.api.Assertions.assertEquals(7, r.nIter,
				"Expected scipy-parity iteration count (7) but got " + r.nIter);
	}
}
