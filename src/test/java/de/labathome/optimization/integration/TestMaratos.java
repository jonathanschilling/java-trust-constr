package de.labathome.optimization.integration;

import java.util.function.Function;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.MinimizeTrustConstr;
import org.scipy.optimize.minimize.NonlinearConstraint;
import org.scipy.optimize.minimize.records.OptimizeResult;
import org.ujmp.core.Matrix;

import minerva.tests.junit.MinervaAssertions;

/**
 * Maratos test problem (Nocedal &amp; Wright, problem 15.4) — the canonical
 * trust-constr stress-test for the equality-constrained SQP path.
 *
 * <pre>
 *     minimize  2*(x[0]^2 + x[1]^2 - 1) - x[0]
 *     s.t.      x[0]^2 + x[1]^2 = 1
 * </pre>
 *
 * Solution: {@code x* = (1, 0)}, {@code f* = -1}. The unit-circle constraint
 * is nonlinear, so this exercises {@link NonlinearConstraint} in addition to
 * the SQP loop. Mirrors {@code Maratos} in
 * {@code scipy/optimize/tests/test_minimize_constrained.py}.
 */
class TestMaratos {

	private static final double TOL = 1.0e-5;

	@Test
	void maratosFromArc60Degrees() {
		Function<Matrix, Double> fun = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return 2.0 * (a * a + b * b - 1.0) - a;
		};
		Function<Matrix, Matrix> grad = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[] { 4.0 * a - 1.0, 4.0 * b });
		};
		Function<Matrix, Matrix> hess = x ->
				Matrix.Factory.linkToArray(new double[][] { {4.0, 0.0}, {0.0, 4.0} });

		// Constraint: c(x) = x[0]^2 + x[1]^2; require c(x) == 1.
		Function<Matrix, Matrix> constrFun = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[] { a * a + b * b });
		};
		Function<Matrix, Matrix> constrJac = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[][] { {2.0 * a, 2.0 * b} });
		};
		// Hessian of v[0] * c(x) = v[0] * (x[0]^2 + x[1]^2) is 2 * v[0] * I.
		java.util.function.BiFunction<Matrix, Matrix, Matrix> constrHess = (x, v) -> {
			double v0 = v.getAsDouble(0, 0);
			return Matrix.Factory.linkToArray(new double[][] {
					{2.0 * v0, 0.0}, {0.0, 2.0 * v0} });
		};
		NonlinearConstraint c = new NonlinearConstraint(constrFun, constrJac, constrHess,
				new double[] {1.0}, new double[] {1.0}, null);

		// scipy starts at angle 60°: x0 = (cos 60°, sin 60°) = (0.5, sqrt(3)/2). The
		// SOC + analytic constraint Hessian let the algorithm converge in 5 iterations
		// (scipy converges in 8 from the same start).
		double rad = Math.PI / 3;  // 60°
		Matrix x0 = Matrix.Factory.linkToArray(new double[] {Math.cos(rad), Math.sin(rad)});
		OptimizeResult r = MinimizeTrustConstr.minimize(fun, grad, hess, x0, c,
				100, 1.0e-10, 1.0e-10);

		Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), TOL);
		Assertions.assertEquals(0.0, r.x.getAsDouble(1, 0), TOL);
		MinervaAssertions.assertRelAbsEquals(-1.0, r.fun, TOL);
	}
}
