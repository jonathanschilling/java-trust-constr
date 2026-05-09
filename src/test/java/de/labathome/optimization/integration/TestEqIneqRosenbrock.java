package de.labathome.optimization.integration;

import java.util.function.Function;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.LinearConstraint;
import de.labathome.trustconstr.MinimizeTrustConstr;
import de.labathome.trustconstr.records.OptimizeResult;
import de.labathome.trustconstr.matrix.Matrix;

/**
 * Rosenbrock with combined equality + inequality linear constraints, from
 * scipy's {@code test_minimize_constrained.py::EqIneqRosenbrock}.
 *
 * <pre>
 *     minimize  100*(y - x^2)^2 + (1 - x)^2
 *     s.t.  x + 2y &lt;= 1
 *           2x + y == 1
 * </pre>
 *
 * Optimum: {@code x* ~ (0.41494, 0.17011)}.
 *
 * <p>Exercises the multi-constraint dispatch with mixed linear-equality and
 * linear-inequality rows, on a nonconvex objective (Rosenbrock).
 */
class TestEqIneqRosenbrock {

	@Test
	void rosenbrockWithEqAndIneq() {
		Function<Matrix, Double> rosen = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return 100.0 * (b - a * a) * (b - a * a) + (1.0 - a) * (1.0 - a);
		};
		Function<Matrix, Matrix> rosenG = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[] {
					-2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
					200.0 * (b - a * a) });
		};
		Function<Matrix, Matrix> rosenH = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[][] {
					{ 2.0 - 400.0 * b + 1200.0 * a * a, -400.0 * a },
					{ -400.0 * a, 200.0 } });
		};

		LinearConstraint ineq = new LinearConstraint(
				Matrix.Factory.linkToArray(new double[][] { {1.0, 2.0} }),
				new double[] {Double.NEGATIVE_INFINITY}, new double[] {1.0});
		LinearConstraint eq = new LinearConstraint(
				Matrix.Factory.linkToArray(new double[][] { {2.0, 1.0} }),
				new double[] {1.0}, new double[] {1.0});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {-1.0, -0.5});
		// Project x0 to satisfy 2x + y = 1 -- interior-point methods need a strictly
		// feasible start with respect to the equality constraint as well.
		// 2*(-1) + (-0.5) = -2.5; offset (1 - (-2.5))/||(2,1)||^2 = 3.5/5 = 0.7.
		// shift by 0.7 * (2, 1) = (1.4, 0.7) -> x0_proj = (0.4, 0.2). Check: 2*0.4 + 0.2 = 1 OK
		Matrix x0Feas = Matrix.Factory.linkToArray(new double[] {0.4, 0.2});
		OptimizeResult r = MinimizeTrustConstr.minimize(rosen, rosenG, rosenH, x0Feas,
				new Object[] {eq, ineq}, 2000, 1.0e-8, 1.0e-8);

		Assertions.assertEquals(0.41494, r.x.getAsDouble(0, 0), 1.0e-2);
		Assertions.assertEquals(0.17011, r.x.getAsDouble(1, 0), 1.0e-2);
		// KKT at the optimum: g + Jeq^T v + Jineq^T lambda = 0. Mixed eq/ineq
		// exercises both halves of the augmented-system multiplier slice.
		Assertions.assertNotNull(r.lagrangianGrad);
		Assertions.assertTrue(r.lagrangianGrad.normInf() < 1.0e-3,
				"Lagrangian gradient too large: " + r.lagrangianGrad.normInf());
	}

	@Test
	void rosenbrockWithIneqOnly() {
		// scipy::IneqRosenbrock -- minimize Rosenbrock s.t. x[0] + 2 x[1] <= 1.
		// Optimum: ~ (0.5022, 0.2489).
		Function<Matrix, Double> rosen = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return 100.0 * (b - a * a) * (b - a * a) + (1.0 - a) * (1.0 - a);
		};
		Function<Matrix, Matrix> rosenG = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[] {
					-2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
					200.0 * (b - a * a) });
		};
		Function<Matrix, Matrix> rosenH = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[][] {
					{ 2.0 - 400.0 * b + 1200.0 * a * a, -400.0 * a },
					{ -400.0 * a, 200.0 } });
		};

		LinearConstraint ineq = new LinearConstraint(
				Matrix.Factory.linkToArray(new double[][] { {1.0, 2.0} }),
				new double[] {Double.NEGATIVE_INFINITY}, new double[] {1.0});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {-1.0, -0.5});
		OptimizeResult r = MinimizeTrustConstr.minimize(rosen, rosenG, rosenH, x0, ineq,
				2000, 1.0e-8, 1.0e-8);

		Assertions.assertEquals(0.5022, r.x.getAsDouble(0, 0), 1.0e-2);
		Assertions.assertEquals(0.2489, r.x.getAsDouble(1, 0), 1.0e-2);
	}
}
