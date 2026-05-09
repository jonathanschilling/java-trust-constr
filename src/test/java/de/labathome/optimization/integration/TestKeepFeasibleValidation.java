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
 * Tests the {@code keep_feasible=true} strict-feasibility precondition.
 *
 * <p>Scipy's contract: a constraint row marked {@code keep_feasible=true}
 * must already be satisfied at the supplied {@code x0}, because the
 * algorithm will not repair an infeasible kf row mid-run. Our orchestrator
 * validates this at every public dispatch entry and throws
 * {@link IllegalArgumentException} on the first violation.
 */
class TestKeepFeasibleValidation {

	private static Function<Matrix, Double> rosen() {
		return x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return (1 - a) * (1 - a) + 100.0 * (b - a * a) * (b - a * a);
		};
	}

	private static Function<Matrix, Matrix> rosenG() {
		return x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[] {
					-2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
					200.0 * (b - a * a)});
		};
	}

	private static Function<Matrix, Matrix> rosenH() {
		return x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[][] {
					{2.0 - 400.0 * b + 1200.0 * a * a, -400.0 * a},
					{-400.0 * a, 200.0}});
		};
	}

	@Test
	void linearConstraintRejectsInfeasibleKeepFeasibleStart() {
		// Constraint: x[0] >= 5 with keep_feasible=true. x0 = (0.5, 1.5)
		// gives A x[0] = 0.5 < 5 — infeasible. Should throw.
		Matrix A = Matrix.Factory.linkToArray(new double[][] {{1.0, 0.0}});
		LinearConstraint ineq = new LinearConstraint(A,
				new double[] {5.0}, new double[] {Double.POSITIVE_INFINITY},
				new boolean[] {true});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.5});
		IllegalArgumentException ex = Assertions.assertThrows(
				IllegalArgumentException.class,
				() -> MinimizeTrustConstr.minimize(rosen(), rosenG(), rosenH(), x0, ineq,
						100, 1.0e-6, 1.0e-6));
		Assertions.assertTrue(ex.getMessage().contains("keep_feasible"),
				"Error should mention keep_feasible; got: " + ex.getMessage());
	}

	@Test
	void linearConstraintAcceptsFeasibleKeepFeasibleStart() {
		// Same constraint, but x0 = (10, 1.5) satisfies x[0] >= 5.
		Matrix A = Matrix.Factory.linkToArray(new double[][] {{1.0, 0.0}});
		LinearConstraint ineq = new LinearConstraint(A,
				new double[] {5.0}, new double[] {Double.POSITIVE_INFINITY},
				new boolean[] {true});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {10.0, 1.5});
		OptimizeResult r = Assertions.assertDoesNotThrow(() ->
				MinimizeTrustConstr.minimize(rosen(), rosenG(), rosenH(), x0, ineq,
						1000, 1.0e-8, 1.0e-8));
		// Constrained optimum: x[0] = 5, x[1] = 25.
		Assertions.assertEquals(5.0, r.x.getAsDouble(0, 0), 1.0e-3);
	}

	@Test
	void nonlinearConstraintRejectsInfeasibleKeepFeasibleStart() {
		// Constraint: x[0]^2 + x[1]^2 <= 1 with keep_feasible=true. x0 = (3, 0)
		// gives 9 > 1 — infeasible.
		Function<Matrix, Matrix> cFun = x -> Matrix.Factory.linkToArray(new double[] {
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0)
				+ x.getAsDouble(1, 0) * x.getAsDouble(1, 0)});
		Function<Matrix, Matrix> cJac = x -> Matrix.Factory.linkToArray(new double[][] {
				{2 * x.getAsDouble(0, 0), 2 * x.getAsDouble(1, 0)}});
		NonlinearConstraint c = new NonlinearConstraint(cFun, cJac, null,
				new double[] {Double.NEGATIVE_INFINITY}, new double[] {1.0},
				new boolean[] {true});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {3.0, 0.0});
		IllegalArgumentException ex = Assertions.assertThrows(
				IllegalArgumentException.class,
				() -> MinimizeTrustConstr.minimize(rosen(), rosenG(), rosenH(), x0, c,
						100, 1.0e-6, 1.0e-6));
		Assertions.assertTrue(ex.getMessage().contains("keep_feasible"),
				"Error should mention keep_feasible; got: " + ex.getMessage());
	}

	@Test
	void keepFeasibleFalseDoesNotRequireFeasibleStart() {
		// Same as the rejecting case but with keep_feasible=false (default).
		// IP path tolerates infeasible kf=false starts.
		Matrix A = Matrix.Factory.linkToArray(new double[][] {{1.0, 0.0}});
		LinearConstraint ineq = new LinearConstraint(A,
				new double[] {-2.0}, new double[] {Double.POSITIVE_INFINITY},
				new boolean[] {false});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.5});
		OptimizeResult r = Assertions.assertDoesNotThrow(() ->
				MinimizeTrustConstr.minimize(rosen(), rosenG(), rosenH(), x0, ineq,
						1000, 1.0e-8, 1.0e-8));
		Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 1.0e-3);
	}

	@Test
	void keepFeasibleFlowsToIpEnforceFeasibility() {
		// Active linear-inequality with keep_feasible=true. The orchestrator
		// extracts the per-canonical-ineq kf flags and passes them as
		// `enforceFeasibility` to TrustRegionInteriorPoint, which in turn
		// uses them in BarrierSubproblem.computeFunction to drive the slack
		// to maintain feasibility. The test verifies the algorithm still
		// converges to the active-bound optimum.
		Matrix A = Matrix.Factory.linkToArray(new double[][] {{1.0, 0.0}});
		LinearConstraint ineq = new LinearConstraint(A,
				new double[] {2.0}, new double[] {Double.POSITIVE_INFINITY},
				new boolean[] {true});

		// Feasible start (matches the kf=true validator).
		Matrix x0 = Matrix.Factory.linkToArray(new double[] {3.0, 9.0});
		OptimizeResult r = MinimizeTrustConstr.minimize(rosen(), rosenG(), rosenH(), x0, ineq,
				1000, 1.0e-8, 1.0e-8);
		Assertions.assertEquals(2.0, r.x.getAsDouble(0, 0), 1.0e-3);
		Assertions.assertEquals(4.0, r.x.getAsDouble(1, 0), 1.0e-2);
		Assertions.assertTrue(r.success, "Should converge with kf=true");
	}

	@Test
	void enforceFeasibilityIneqExtractsKfFromCanonicalRows() {
		// Unit test for the LinearConstraint.enforceFeasibilityIneq() helper.
		// Two-sided interval row with kf=true should produce TWO canonical
		// ineq rows (ub then lb), both with kf=true.
		Matrix A = Matrix.Factory.linkToArray(new double[][] {{1.0}});
		LinearConstraint c = new LinearConstraint(A,
				new double[] {-1.0}, new double[] {1.0},
				new boolean[] {true});

		Assertions.assertEquals(2, c.nIneq());
		boolean[] kf = c.enforceFeasibilityIneq();
		Assertions.assertEquals(2, kf.length);
		Assertions.assertTrue(kf[0], "ub-row inherits kf");
		Assertions.assertTrue(kf[1], "lb-row inherits kf");
	}

	@Test
	void multiConstraintValidatesEachSource() {
		// Two constraints; only the second has keep_feasible=true and is
		// infeasible at x0. Validator should catch the violation despite
		// the first being fine.
		LinearConstraint ok = new LinearConstraint(
				Matrix.Factory.linkToArray(new double[][] {{1.0, 1.0}}),
				new double[] {Double.NEGATIVE_INFINITY}, new double[] {10.0},
				new boolean[] {false});
		LinearConstraint kf = new LinearConstraint(
				Matrix.Factory.linkToArray(new double[][] {{0.0, 1.0}}),
				new double[] {5.0}, new double[] {Double.POSITIVE_INFINITY},
				new boolean[] {true});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.5});
		Assertions.assertThrows(IllegalArgumentException.class,
				() -> MinimizeTrustConstr.minimize(rosen(), rosenG(), rosenH(), x0,
						new Object[] {ok, kf}, 100, 1.0e-6, 1.0e-6));
	}
}
