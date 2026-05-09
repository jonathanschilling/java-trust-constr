package de.labathome.optimization.integration;

import java.util.function.BiFunction;
import java.util.function.Function;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.MinimizeTrustConstr;
import org.scipy.optimize.minimize.NonlinearConstraint;
import org.scipy.optimize.minimize.records.OptimizeResult;
import org.scipy.optimize.minimize.matrix.Matrix;

/**
 * Port of scipy {@code TestEmptyConstraint::test_empty_constraint}:
 *
 * <pre>
 *     minimize  x^2 + y^2
 *     s.t.      x^2 - y^2 &gt;= 1
 * </pre>
 *
 * The unconstrained minimum (0, 0) is infeasible. The constrained optimum lies
 * on the boundary of the unit hyperbola, {@code x = +/-1, y = 0}, with
 * {@code f = 1}.
 *
 * <p>Exercises the analytic constraint Hessian-of-Lagrangian path with a
 * non-convex constraint set.
 */
class TestUnitHyperbola {

	@Test
	void quadraticOutsideUnitHyperbola() {
		Function<Matrix, Double> fun = x ->
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0) + x.getAsDouble(1, 0) * x.getAsDouble(1, 0);
		Function<Matrix, Matrix> grad = x ->
				Matrix.Factory.linkToArray(new double[] {
						2 * x.getAsDouble(0, 0),
						2 * x.getAsDouble(1, 0) });
		Function<Matrix, Matrix> hess = x ->
				Matrix.Factory.linkToArray(new double[][] { {2, 0}, {0, 2} });

		Function<Matrix, Matrix> cFun = x -> Matrix.Factory.linkToArray(new double[] {
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0) - x.getAsDouble(1, 0) * x.getAsDouble(1, 0)
		});
		Function<Matrix, Matrix> cJac = x -> Matrix.Factory.linkToArray(new double[][] {
				{ 2 * x.getAsDouble(0, 0), -2 * x.getAsDouble(1, 0) }
		});
		// Hessian-of-Lagrangian for v[0] * (x^2 - y^2) is v[0] * diag(2, -2).
		BiFunction<Matrix, Matrix, Matrix> cHess = (x, v) -> {
			double v0 = v.getAsDouble(0, 0);
			return Matrix.Factory.linkToArray(new double[][] {
					{2 * v0, 0}, {0, -2 * v0} });
		};

		NonlinearConstraint c = new NonlinearConstraint(cFun, cJac, cHess,
				new double[] {1.0}, new double[] {Double.POSITIVE_INFINITY}, null);

		// Start at scipy's (1, 2). Already satisfies x^2 - y^2 = 1 - 4 = -3 -- INFEASIBLE.
		// IP needs a strictly feasible start, so start at a point that satisfies it.
		// (1.5, 0): 2.25 - 0 = 2.25 > 1 OK
		Matrix x0 = Matrix.Factory.linkToArray(new double[] {1.5, 0.0});
		OptimizeResult r = MinimizeTrustConstr.minimize(fun, grad, hess, x0, c,
				1000, 1.0e-8, 1.0e-8);

		// Boundary optimum: |x| = 1, y = 0, f = 1.
		Assertions.assertEquals(1.0, Math.abs(r.x.getAsDouble(0, 0)), 1.0e-3);
		Assertions.assertEquals(0.0, r.x.getAsDouble(1, 0), 1.0e-3);
		Assertions.assertEquals(1.0, r.fun, 1.0e-3);
		// KKT at the boundary optimum: g + Jineq^T lambda = 0, with constraint
		// active and lambda != 0. Sharpest test of multiplier recovery.
		Assertions.assertNotNull(r.lagrangianGrad);
		Assertions.assertTrue(r.lagrangianGrad.normInf() < 1.0e-3,
				"Lagrangian gradient too large: " + r.lagrangianGrad.normInf());
	}
}
