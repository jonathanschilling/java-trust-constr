package de.labathome.optimization.integration;

import java.util.function.Function;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.LinearConstraint;
import org.scipy.optimize.minimize.MinimizeTrustConstr;
import org.scipy.optimize.minimize.SR1;
import org.scipy.optimize.minimize.records.OptimizeResult;
import org.scipy.optimize.minimize.matrix.Matrix;

/**
 * Tests for the quasi-Newton (BFGS / SR1) and finite-difference overloads of
 * {@code MinimizeTrustConstr.minimize}. The orchestrator builds a fresh
 * Hessian-update strategy when the user omits {@code hess}, and 2-point
 * finite-difference gradient estimation when {@code grad} is also omitted —
 * mirroring scipy's defaults.
 */
class TestQuasiNewtonAndFiniteDiff {

	@Test
	void quadraticOnHyperplaneWithBFGS() {
		// minimize x[0]^2 + x[1]^2  s.t.  x[0] + x[1] = 2  ->  (1, 1), f = 2.
		Function<Matrix, Double> q = x ->
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0) + x.getAsDouble(1, 0) * x.getAsDouble(1, 0);
		Function<Matrix, Matrix> qg = x ->
				Matrix.Factory.linkToArray(new double[] {2 * x.getAsDouble(0, 0), 2 * x.getAsDouble(1, 0)});

		LinearConstraint eq = new LinearConstraint(
				Matrix.Factory.linkToArray(new double[][] { {1.0, 1.0} }),
				new double[] {2.0}, new double[] {2.0});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {-1.0, 0.5});
		OptimizeResult r = MinimizeTrustConstr.minimize(q, qg, x0, eq,
				200, 1.0e-8, 1.0e-8);

		Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 1.0e-5);
		Assertions.assertEquals(1.0, r.x.getAsDouble(1, 0), 1.0e-5);
		Assertions.assertEquals(2.0, r.fun, 1.0e-5);
	}

	@Test
	void quadraticOnHyperplaneWithFiniteDiffAndBFGS() {
		// Most hands-off entry: only the objective is supplied. Gradient comes
		// from 2-point finite differences; Hessian from BFGS.
		Function<Matrix, Double> q = x ->
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0) + x.getAsDouble(1, 0) * x.getAsDouble(1, 0);

		LinearConstraint eq = new LinearConstraint(
				Matrix.Factory.linkToArray(new double[][] { {1.0, 1.0} }),
				new double[] {2.0}, new double[] {2.0});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {-1.0, 0.5});
		OptimizeResult r = MinimizeTrustConstr.minimize(q, x0, eq, 200, 1.0e-8, 1.0e-8);

		Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 1.0e-3);
		Assertions.assertEquals(1.0, r.x.getAsDouble(1, 0), 1.0e-3);
		Assertions.assertEquals(2.0, r.fun, 1.0e-3);
	}

	@Test
	void rosenbrockOnHyperplaneWithSR1() {
		// SR1 is better-suited to nonconvex objectives like Rosenbrock than to
		// convex quadratics (where its curvature condition often fails).
		Function<Matrix, Double> rosen = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return (1 - a) * (1 - a) + 100.0 * (b - a * a) * (b - a * a);
		};
		Function<Matrix, Matrix> rosenG = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[] {
					-2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
					200.0 * (b - a * a) });
		};
		LinearConstraint eq = new LinearConstraint(
				Matrix.Factory.linkToArray(new double[][] { {1.0, 1.0} }),
				new double[] {2.0}, new double[] {2.0});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.5});
		OptimizeResult r = MinimizeTrustConstr.minimize(rosen, rosenG, SR1.FACTORY.build(), x0, eq,
				1000, 1.0e-8, 1.0e-8);

		Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 1.0e-2);
		Assertions.assertEquals(1.0, r.x.getAsDouble(1, 0), 1.0e-2);
	}

	@Test
	void rosenbrockOnHyperplaneWithBFGS() {
		// Rosenbrock with x[0] + x[1] = 2; optimum (1, 1), f = 0. With BFGS the
		// algorithm converges (slower than analytic but reliable).
		Function<Matrix, Double> rosen = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return (1 - a) * (1 - a) + 100.0 * (b - a * a) * (b - a * a);
		};
		Function<Matrix, Matrix> rosenG = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[] {
					-2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
					200.0 * (b - a * a) });
		};
		LinearConstraint eq = new LinearConstraint(
				Matrix.Factory.linkToArray(new double[][] { {1.0, 1.0} }),
				new double[] {2.0}, new double[] {2.0});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.5});
		OptimizeResult r = MinimizeTrustConstr.minimize(rosen, rosenG, x0, eq,
				500, 1.0e-8, 1.0e-8);

		Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 1.0e-3);
		Assertions.assertEquals(1.0, r.x.getAsDouble(1, 0), 1.0e-3);
		Assertions.assertTrue(r.fun < 1.0e-4, "f should be tiny but was " + r.fun);
	}

	@Test
	void unconstrainedRosenbrockWithBFGS() {
		// Pure unconstrained Rosenbrock through trust-constr with BFGS.
		// Routes through the equality SQP with empty constraint set; the
		// BFGS strategy supplies the curvature.
		Function<Matrix, Double> rosen = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return (1 - a) * (1 - a) + 100.0 * (b - a * a) * (b - a * a);
		};
		Function<Matrix, Matrix> rosenG = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[] {
					-2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
					200.0 * (b - a * a)});
		};

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.5});
		LinearConstraint noConstraint = null;
		OptimizeResult r = MinimizeTrustConstr.minimize(rosen, rosenG, x0, noConstraint,
				500, 1.0e-8, 1.0e-8);

		Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 1.0e-3);
		Assertions.assertEquals(1.0, r.x.getAsDouble(1, 0), 1.0e-3);
		Assertions.assertTrue(r.fun < 1.0e-4, "f should be tiny but was " + r.fun);
	}

	@Test
	void rosenbrockUnderActiveBoundWithBFGS() {
		// Active-bound Rosenbrock (x[0] >= 2) through the IP path with BFGS
		// providing the Hessian. Active-constraint dispatch under quasi-Newton
		// is one cell of scipy's TestTrustRegionConstr parametric matrix.
		Function<Matrix, Double> rosen = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return (1 - a) * (1 - a) + 100.0 * (b - a * a) * (b - a * a);
		};
		Function<Matrix, Matrix> rosenG = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[] {
					-2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
					200.0 * (b - a * a)});
		};

		Matrix A = Matrix.Factory.linkToArray(new double[][] {{1.0, 0.0}});
		LinearConstraint ineq = new LinearConstraint(A,
				new double[] {2.0}, new double[] {Double.POSITIVE_INFINITY});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {3.0, 9.0});
		OptimizeResult r = MinimizeTrustConstr.minimize(rosen, rosenG, x0, ineq,
				1000, 1.0e-8, 1.0e-8);

		Assertions.assertEquals(2.0, r.x.getAsDouble(0, 0), 1.0e-3);
		Assertions.assertEquals(4.0, r.x.getAsDouble(1, 0), 1.0e-2);
	}

	@Test
	void unitHyperbolaWithBFGSForObjectiveAnalyticConstraintHessian() {
		// Boundary optimum on a non-convex constraint (x^2 - y^2 >= 1) with
		// BFGS approximating the (constant 2I) objective Hessian and the
		// constraint Hessian supplied analytically. Mirrors scipy's pattern of
		// mixing analytic constraint Hessian with quasi-Newton on the objective.
		Function<Matrix, Double> fun = x ->
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0)
				+ x.getAsDouble(1, 0) * x.getAsDouble(1, 0);
		Function<Matrix, Matrix> grad = x ->
				Matrix.Factory.linkToArray(new double[] {
						2 * x.getAsDouble(0, 0),
						2 * x.getAsDouble(1, 0)});

		Function<Matrix, Matrix> cFun = x -> Matrix.Factory.linkToArray(new double[] {
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0)
				- x.getAsDouble(1, 0) * x.getAsDouble(1, 0)});
		Function<Matrix, Matrix> cJac = x -> Matrix.Factory.linkToArray(new double[][] {
				{2 * x.getAsDouble(0, 0), -2 * x.getAsDouble(1, 0)}});
		java.util.function.BiFunction<Matrix, Matrix, Matrix> cHess = (x, v) -> {
			double v0 = v.getAsDouble(0, 0);
			return Matrix.Factory.linkToArray(new double[][] {
					{2 * v0, 0}, {0, -2 * v0}});
		};
		org.scipy.optimize.minimize.NonlinearConstraint c =
				new org.scipy.optimize.minimize.NonlinearConstraint(
						cFun, cJac, cHess,
						new double[] {1.0},
						new double[] {Double.POSITIVE_INFINITY},
						null);

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {1.5, 0.0});
		// No-Hessian overload: orchestrator builds a fresh BFGS strategy for
		// the objective Hessian and combines it with c.lagrangianContribution
		// to form the full Lagrangian Hessian.
		OptimizeResult r = MinimizeTrustConstr.minimize(fun, grad, x0, c,
				1000, 1.0e-8, 1.0e-8);

		Assertions.assertEquals(1.0, Math.abs(r.x.getAsDouble(0, 0)), 5.0e-2);
		Assertions.assertEquals(0.0, r.x.getAsDouble(1, 0), 5.0e-2);
		Assertions.assertEquals(1.0, r.fun, 1.0e-2);
	}

	@Test
	void rosenbrockUnderInactiveBoundWithFiniteDiffAndBFGS() {
		// Hands-off entry on an inequality-constrained problem: only the
		// objective is supplied. Gradient via 2-point finite differences,
		// Hessian via BFGS, dispatch through the IP path. Mirrors the
		// "{grad: '2-point', hess: BFGS}" cell of scipy's parametric matrix.
		Function<Matrix, Double> rosen = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return (1 - a) * (1 - a) + 100.0 * (b - a * a) * (b - a * a);
		};

		Matrix A = Matrix.Factory.linkToArray(new double[][] {{1.0, 0.0}});
		LinearConstraint ineq = new LinearConstraint(A,
				new double[] {-2.0}, new double[] {Double.POSITIVE_INFINITY});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.5});
		OptimizeResult r = MinimizeTrustConstr.minimize(rosen, x0, ineq,
				2000, 1.0e-6, 1.0e-6);

		// FD gradient is noisier than analytic, so loosen tolerance.
		Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 1.0e-2);
		Assertions.assertEquals(1.0, r.x.getAsDouble(1, 0), 1.0e-2);
		Assertions.assertTrue(r.fun < 1.0e-3, "f should be tiny but was " + r.fun);
	}
}
