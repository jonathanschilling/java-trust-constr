package de.labathome.optimization.integration;

import java.util.function.Function;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.LinearConstraint;
import org.scipy.optimize.minimize.MinimizeTrustConstr;
import org.scipy.optimize.minimize.records.OptimizeResult;
import org.scipy.optimize.minimize.matrix.Matrix;

/**
 * End-to-end cases ported from
 * {@code scipy/optimize/tests/test_minimize_constrained.py} that aren't already
 * covered by {@code TestMaratos}, {@code TestHyperbolicIneq},
 * {@code TestEqualityConstrainedRosenbrock}, {@code TestEqIneqRosenbrock},
 * {@code TestBoundedRosenbrock}, or {@code TestElec}:
 *
 * <ul>
 *   <li>{@code TestTrustRegionConstr::test_no_constraints} -- unconstrained
 *       Rosenbrock through trust-constr (routes through the equality path with
 *       an empty constraint set; needs the zero-row shortcuts in
 *       {@code Projections.projections} and the {@code safeNorm2} guards in
 *       {@code EqualityConstrainedSQP}).</li>
 *   <li>{@code test_bug_11886} -- quadratic with a one-sided lower-bound linear
 *       inequality; smoke check that the orchestrator doesn't error.</li>
 * </ul>
 */
class TestMinimizeConstrainedExtras {

	@Test
	void unconstrainedRosenbrock() {
		// Scipy's TestTrustRegionConstr::test_no_constraints -- unconstrained
		// Rosenbrock should converge to (1, 1) through trust-constr just like
		// L-BFGS-B does. Routes through the equality path with an empty
		// constraint set; the Projections.projections zero-row shortcut plus
		// EqualityConstrainedSQP.safeNorm2 keep the SQP loop's normal-step /
		// merit-function arithmetic well-defined when there are no rows of b.
		Function<Matrix, Double> rosen = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return (1.0 - a) * (1.0 - a) + 100.0 * (b - a * a) * (b - a * a);
		};
		Function<Matrix, Matrix> rosenG = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[] {
					-2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
					200.0 * (b - a * a)});
		};
		Function<Matrix, Matrix> rosenH = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[][] {
					{2.0 - 400.0 * b + 1200.0 * a * a, -400.0 * a},
					{-400.0 * a, 200.0}});
		};

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.5});
		LinearConstraint noConstraint = null;
		OptimizeResult r = MinimizeTrustConstr.minimize(rosen, rosenG, rosenH, x0,
				noConstraint, 1000, 1.0e-10, 1.0e-10);

		Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 1.0e-5);
		Assertions.assertEquals(1.0, r.x.getAsDouble(1, 0), 1.0e-5);
		Assertions.assertEquals(0.0, r.fun, 1.0e-10);
		// Lagrangian gradient at the unconstrained optimum reduces to the
		// ordinary gradient, which vanishes.
		Assertions.assertNotNull(r.lagrangianGrad);
		Assertions.assertTrue(r.lagrangianGrad.normInf() < 1.0e-6,
				"Lagrangian gradient too large: " + r.lagrangianGrad.normInf());
	}

	@Test
	void quadraticOneSidedLinearLowerBound() {
		// Scipy's test_bug_11886: minimize x[0]^2 + x[1]^2 s.t. x[i] >= -1.
		// The unconstrained optimum (0, 0) is in the feasible interior, so
		// the solver should converge to the origin without the bound binding.
		// Originally a smoke test -- we additionally check the result.
		Function<Matrix, Double> q = x ->
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0)
				+ x.getAsDouble(1, 0) * x.getAsDouble(1, 0);
		Function<Matrix, Matrix> qg = x ->
				Matrix.Factory.linkToArray(new double[] {
						2 * x.getAsDouble(0, 0),
						2 * x.getAsDouble(1, 0)});
		Function<Matrix, Matrix> qh = x ->
				Matrix.Factory.linkToArray(new double[][] {{2, 0}, {0, 2}});

		// Identity Jacobian, lower bound -1, no upper bound (one-sided).
		Matrix A = Matrix.Factory.eye(2, 2);
		LinearConstraint ineq = new LinearConstraint(A,
				new double[] {-1.0, -1.0},
				new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {1.0, 1.0});
		OptimizeResult r = Assertions.assertDoesNotThrow(() ->
				MinimizeTrustConstr.minimize(q, qg, qh, x0, ineq,
						200, 1.0e-8, 1.0e-8));

		Assertions.assertEquals(0.0, r.x.getAsDouble(0, 0), 1.0e-4);
		Assertions.assertEquals(0.0, r.x.getAsDouble(1, 0), 1.0e-4);
		Assertions.assertEquals(0.0, r.fun, 1.0e-7);
	}

	@Test
	void degenerateConstraintReportsFailure() {
		// Port of scipy test_issue_18882: minimize u1^2 + u2^2 subject to
		// 1 + u1^2/9 - u2^2/16 = 0, starting at (0, 0). At the start, the
		// constraint is infeasible (= 1) and its Jacobian is (0, 0) -- singular.
		// Scipy: success=False AND constr_violation > 1e-8.
		java.util.function.Function<Matrix, Double> obj = u ->
				u.getAsDouble(0, 0) * u.getAsDouble(0, 0)
				+ u.getAsDouble(1, 0) * u.getAsDouble(1, 0);
		java.util.function.Function<Matrix, Matrix> objGrad = u ->
				Matrix.Factory.linkToArray(new double[] {
						2 * u.getAsDouble(0, 0),
						2 * u.getAsDouble(1, 0)});
		java.util.function.Function<Matrix, Matrix> objHess = u ->
				Matrix.Factory.linkToArray(new double[][] {{2, 0}, {0, 2}});

		java.util.function.Function<Matrix, Matrix> cFun = u -> {
			double u1 = u.getAsDouble(0, 0);
			double u2 = u.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[] {
					1.0 + u1 * u1 / 9.0 - u2 * u2 / 16.0});
		};
		java.util.function.Function<Matrix, Matrix> cJac = u ->
				Matrix.Factory.linkToArray(new double[][] {
						{2.0 * u.getAsDouble(0, 0) / 9.0,
						 -2.0 * u.getAsDouble(1, 0) / 16.0}});

		org.scipy.optimize.minimize.NonlinearConstraint c =
				new org.scipy.optimize.minimize.NonlinearConstraint(cFun, cJac,
						new double[] {0.0}, new double[] {0.0});

		Matrix u0 = Matrix.Factory.linkToArray(new double[] {0.0, 0.0});
		// Should not throw -- the algorithm should report failure rather than
		// crash on the singular Jacobian at start.
		OptimizeResult r = Assertions.assertDoesNotThrow(() ->
				MinimizeTrustConstr.minimize(obj, objGrad, objHess, u0, c,
						200, 1.0e-8, 1.0e-8));

		Assertions.assertFalse(r.success,
				"Degenerate constraint should report success=false; got status="
						+ r.status + " cv=" + r.constraintViolation);
		Assertions.assertTrue(r.constraintViolation > 1.0e-8,
				"Constraint violation should be appreciable; got " + r.constraintViolation);
	}

	@Test
	void overdeterminedEqualityRejectedWithHelpfulMessage() {
		// Port of scipy test_gh20665_too_many_constraints: 3 equality
		// constraints in 2 variables. Default (QR) factorization can't
		// handle this -- orchestrator should throw a clear error before the
		// algorithm hits an internal array-bounds failure.
		Matrix A = Matrix.Factory.linkToArray(new double[][] {
				{0, 1}, {2, 3}, {4, 5}});
		LinearConstraint eq = new LinearConstraint(A,
				new double[] {1, 1, 1}, new double[] {1, 1, 1});

		java.util.function.Function<Matrix, Double> rosen = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return (1 - a) * (1 - a) + 100.0 * (b - a * a) * (b - a * a);
		};
		java.util.function.Function<Matrix, Matrix> rosenG = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[] {
					-2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
					200.0 * (b - a * a)});
		};
		java.util.function.Function<Matrix, Matrix> rosenH = x ->
				Matrix.Factory.linkToArray(new double[][] {{2, 0}, {0, 2}});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {1.0, 1.0});
		IllegalArgumentException ex = Assertions.assertThrows(
				IllegalArgumentException.class,
				() -> MinimizeTrustConstr.minimize(rosen, rosenG, rosenH, x0, eq,
						100, 1.0e-6, 1.0e-6));
		Assertions.assertTrue(ex.getMessage().contains("equality constraints")
						&& ex.getMessage().contains("independent variables"),
				"Expected clear over-determined message; got: " + ex.getMessage());
		// Workaround the message advertises: SVD_FACTORIZATION via the full-shape entry.
		Assertions.assertTrue(ex.getMessage().contains("SVD_FACTORIZATION"),
				"Error should suggest SVD_FACTORIZATION as a workaround; got: " + ex.getMessage());
	}
}
