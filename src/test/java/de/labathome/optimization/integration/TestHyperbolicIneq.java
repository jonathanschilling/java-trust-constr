package de.labathome.optimization.integration;

import java.util.function.Function;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.LinearConstraint;
import org.scipy.optimize.minimize.MinimizeTrustConstr;
import org.scipy.optimize.minimize.NonlinearConstraint;
import org.scipy.optimize.minimize.records.Bounds;
import org.scipy.optimize.minimize.records.OptimizeResult;
import org.scipy.optimize.minimize.matrix.Matrix;

/**
 * Problem 15.1 from Nocedal &amp; Wright via scipy's
 * {@code test_minimize_constrained.py::HyperbolicIneq}.
 *
 * <pre>
 *     minimize  (1/2)(x[0] - 2)^2 + (1/2)(x[1] - 1/2)^2
 *     s.t.      1/(x[0] + 1) - x[1] &gt;= 1/4
 *               x[0] &gt;= 0
 *               x[1] &gt;= 0
 * </pre>
 *
 * Optimum: {@code x* ~ (1.952823, 0.088659)}.
 *
 * <p>Exercises the multi-constraint dispatch with one nonlinear inequality
 * (the hyperbola) and bounds-as-constraint (the two non-negativity bounds
 * promoted via {@link LinearConstraint#fromBounds}).
 */
class TestHyperbolicIneq {

	@Test
	void hyperbolicInequalityFromOrigin() {
		Function<Matrix, Double> fun = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return 0.5 * (a - 2) * (a - 2) + 0.5 * (b - 0.5) * (b - 0.5);
		};
		Function<Matrix, Matrix> grad = x ->
				Matrix.Factory.linkToArray(new double[] {
						x.getAsDouble(0, 0) - 2,
						x.getAsDouble(1, 0) - 0.5 });
		Function<Matrix, Matrix> hess = x ->
				Matrix.Factory.linkToArray(new double[][] { {1, 0}, {0, 1} });

		Function<Matrix, Matrix> cFun = x -> Matrix.Factory.linkToArray(new double[] {
				1.0 / (x.getAsDouble(0, 0) + 1.0) - x.getAsDouble(1, 0)
		});
		Function<Matrix, Matrix> cJac = x -> {
			double a = x.getAsDouble(0, 0);
			return Matrix.Factory.linkToArray(new double[][] {
					{ -1.0 / ((a + 1) * (a + 1)), -1.0 }
			});
		};
		// Hessian of v[0] * c(x) = v[0] * (1/(x[0]+1) - x[1]):
		//   d^2/dx[0]^2 = v[0] * 2/(x[0]+1)^3, others zero.
		java.util.function.BiFunction<Matrix, Matrix, Matrix> cHess = (x, v) -> {
			double a = x.getAsDouble(0, 0);
			double v0 = v.getAsDouble(0, 0);
			double h00 = 2.0 * v0 / Math.pow(a + 1.0, 3.0);
			return Matrix.Factory.linkToArray(new double[][] {
					{h00, 0.0}, {0.0, 0.0} });
		};
		NonlinearConstraint hyper = new NonlinearConstraint(cFun, cJac, cHess,
				new double[] {0.25}, new double[] {Double.POSITIVE_INFINITY}, null);

		Bounds nonNeg = new Bounds(
				Matrix.Factory.linkToArray(new double[] {0.0, 0.0}),
				Matrix.Factory.linkToArray(new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY}),
				false);
		LinearConstraint bounds = LinearConstraint.fromBounds(nonNeg);

		// scipy's x0 = (0, 0) is on the boundary of the nonneg bounds -- interior-point
		// methods need a strictly feasible start. Pick a feasible interior point instead.
		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 0.5});
		OptimizeResult r = MinimizeTrustConstr.minimize(fun, grad, hess, x0,
				new Object[] {hyper, bounds}, 2000, 1.0e-8, 1.0e-8);

		Assertions.assertEquals(1.952823, r.x.getAsDouble(0, 0), 1.0e-2);
		Assertions.assertEquals(0.088659, r.x.getAsDouble(1, 0), 1.0e-2);
	}
}
