package de.labathome.optimization.integration;

import java.util.function.Function;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.LinearConstraint;
import org.scipy.optimize.minimize.MinimizeTrustConstr;
import org.scipy.optimize.minimize.NonlinearConstraint;
import org.scipy.optimize.minimize.records.OptimizeResult;
import org.scipy.optimize.minimize.matrix.Matrix;

/**
 * Tests for the multi-constraint overload of
 * {@link MinimizeTrustConstr#minimize(java.util.function.Function,
 * java.util.function.Function, java.util.function.Function, Matrix, Object[],
 * int, double, double)}.
 *
 * <p>Each test passes an array of {@link LinearConstraint} and/or
 * {@link NonlinearConstraint} instances; the orchestrator stitches them via
 * {@code CombinedConstraint} and dispatches as usual.
 */
class TestMultiConstraint {

	@Test
	void twoLinearEqualities() {
		// minimize x[0]^2 + x[1]^2 + x[2]^2
		// s.t.  x[0] + x[1] + x[2] = 6  (equality 1)
		//       x[0] - x[1]        = 0  (equality 2)
		// -> x[0] = x[1] (from eq2); 2*x[0] + x[2] = 6 (from eq1).
		//    Min sum of squares: minimize over x0 then over x2 = 6 - 2*x0:
		//    Lagrangian: 2x0^2 + (6-2x0)^2; deriv = 4x0 - 4(6-2x0) = 12x0 - 24 = 0 -> x0=2.
		//    So x* = (2, 2, 2), f* = 12.
		Function<Matrix, Double> q = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			double c = x.getAsDouble(2, 0);
			return a * a + b * b + c * c;
		};
		Function<Matrix, Matrix> qg = x ->
				Matrix.Factory.linkToArray(new double[] {
						2 * x.getAsDouble(0, 0),
						2 * x.getAsDouble(1, 0),
						2 * x.getAsDouble(2, 0) });
		Function<Matrix, Matrix> qh = x ->
				Matrix.Factory.linkToArray(new double[][] {
						{2, 0, 0}, {0, 2, 0}, {0, 0, 2} });

		LinearConstraint c1 = new LinearConstraint(
				Matrix.Factory.linkToArray(new double[][] { {1, 1, 1} }),
				new double[] {6.0}, new double[] {6.0});
		LinearConstraint c2 = new LinearConstraint(
				Matrix.Factory.linkToArray(new double[][] { {1, -1, 0} }),
				new double[] {0.0}, new double[] {0.0});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.0, 0.0, 0.0});
		OptimizeResult r = MinimizeTrustConstr.minimize(q, qg, qh, x0,
				new Object[] {c1, c2}, 200, 1.0e-10, 1.0e-10);

		Assertions.assertEquals(2.0, r.x.getAsDouble(0, 0), 1e-8);
		Assertions.assertEquals(2.0, r.x.getAsDouble(1, 0), 1e-8);
		Assertions.assertEquals(2.0, r.x.getAsDouble(2, 0), 1e-8);
		Assertions.assertEquals(12.0, r.fun, 1e-8);
	}

	@Test
	void linearEqualityPlusNonlinearInequality() {
		// minimize (x - 2)^2 + (y - 2)^2
		// s.t.  x + y = 1                  (equality)
		//       x^2 + y^2 <= 1             (nonlinear inequality)
		// Project (2, 2) onto x+y=1: (0.5, 0.5). ||(0.5, 0.5)||^2 = 0.5 <= 1, so disk
		// constraint is INACTIVE at the optimum. Optimum: (0.5, 0.5), f = 4.5.
		Function<Matrix, Double> q = x -> {
			double dx = x.getAsDouble(0, 0) - 2.0;
			double dy = x.getAsDouble(1, 0) - 2.0;
			return dx * dx + dy * dy;
		};
		Function<Matrix, Matrix> qg = x ->
				Matrix.Factory.linkToArray(new double[] {
						2 * (x.getAsDouble(0, 0) - 2.0),
						2 * (x.getAsDouble(1, 0) - 2.0) });
		Function<Matrix, Matrix> qh = x ->
				Matrix.Factory.linkToArray(new double[][] { {2, 0}, {0, 2} });

		LinearConstraint eq = new LinearConstraint(
				Matrix.Factory.linkToArray(new double[][] { {1, 1} }),
				new double[] {1.0}, new double[] {1.0});
		Function<Matrix, Matrix> cFun = x -> Matrix.Factory.linkToArray(new double[] {
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0) + x.getAsDouble(1, 0) * x.getAsDouble(1, 0)
		});
		Function<Matrix, Matrix> cJac = x -> Matrix.Factory.linkToArray(new double[][] {
				{ 2 * x.getAsDouble(0, 0), 2 * x.getAsDouble(1, 0) }
		});
		NonlinearConstraint ineq = new NonlinearConstraint(cFun, cJac,
				new double[] {Double.NEGATIVE_INFINITY}, new double[] {1.0});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 0.5});
		OptimizeResult r = MinimizeTrustConstr.minimize(q, qg, qh, x0,
				new Object[] {eq, ineq}, 1000, 1.0e-8, 1.0e-8);

		Assertions.assertEquals(4.5, r.fun, 1e-3);
		double xv = r.x.getAsDouble(0, 0);
		double yv = r.x.getAsDouble(1, 0);
		Assertions.assertEquals(0.5, xv, 1e-3);
		Assertions.assertEquals(0.5, yv, 1e-3);
		Assertions.assertEquals(1.0, xv + yv, 1e-6);
		Assertions.assertTrue(xv * xv + yv * yv <= 1.0 + 1e-6,
				"disk constraint violated: " + (xv*xv + yv*yv));
	}
}
