package de.labathome.optimization.integration;

import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.ToDoubleBiFunction;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.MinimizeTrustConstr;
import org.scipy.optimize.minimize.records.Bounds;
import org.scipy.optimize.minimize.records.OptimizeResult;
import org.scipy.optimize.minimize.matrix.Matrix;

/**
 * Port of scipy {@code test_gh11649}'s structural concern: when
 * {@code Bounds(keep_feasible=True)} is supplied, finite-difference
 * perturbations during gradient evaluation must stay inside the bounds.
 *
 * <p>Method: define an objective that records every {@code x} it's called
 * with. Run trust-constr with FD-grad and bounds. Assert no recorded {@code x}
 * lies outside the bounds.
 *
 * <p>Note: full scipy gh11649 also includes an inequality nonlinear constraint
 * with mid-iteration kf enforcement, which our port doesn't yet provide. This
 * test exercises the FD-bounds-respecting half.
 */
class TestFdRespectsBounds {

	@Test
	void fdPerturbationsRespectBoundsWhenKeepFeasibleTrue() {
		// Track every x at which the objective is evaluated.
		java.util.List<double[]> seen = new java.util.ArrayList<>();
		ToDoubleBiFunction<Matrix, Object> obj = (x, args) -> {
			seen.add(new double[] {x.getAsDouble(0, 0), x.getAsDouble(1, 0)});
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			// Smooth quadratic; doesn't matter for the test.
			return (a - 0.5) * (a - 0.5) + (b - 0.5) * (b - 0.5);
		};

		// Bounds [-1, 1]^2 with keep_feasible=true. Feasible interior start.
		Bounds bnds = new Bounds(
				Matrix.Factory.linkToArray(new double[] {-1.0, -1.0}),
				Matrix.Factory.linkToArray(new double[] {1.0, 1.0}),
				true);

		// Hands-off entry: no analytic grad/hess. The full-shape adapter
		// builds an FD-grad that respects bounds via StrictBounds.
		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.99, -0.99});
		OptimizeResult r = MinimizeTrustConstr.minimizeTrustConstr(
				obj, x0, null,
				null, null,                 // no analytic grad / hess
				null, bnds, null,           // bounds supplied; no other constraints
				1.0e-7, 1.0e-7, 1.0e-7,
				Optional.empty(),
				null,
				500, 0, null,
				1.0, 1.0, 0.1, 0.1,
				null, false);

		// Should converge near (0.5, 0.5) — the unconstrained optimum is in
		// the interior so bounds don't bind.
		Assertions.assertEquals(0.5, r.x.getAsDouble(0, 0), 1.0e-3);
		Assertions.assertEquals(0.5, r.x.getAsDouble(1, 0), 1.0e-3);

		// Critical: every x at which obj was called must lie within bounds.
		for (double[] x : seen) {
			Assertions.assertTrue(x[0] >= -1.0 && x[0] <= 1.0,
					"FD perturbation left x[0] bounds: x=(" + x[0] + ", " + x[1] + ")");
			Assertions.assertTrue(x[1] >= -1.0 && x[1] <= 1.0,
					"FD perturbation left x[1] bounds: x=(" + x[0] + ", " + x[1] + ")");
		}
	}

	@Test
	void boundsKeepFeasibleFlowsToLinearConstraint() {
		// Verify the structural plumbing: Bounds.keepFeasible=true gets
		// translated into per-row keepFeasible[] on the LinearConstraint
		// produced by LinearConstraint.fromBounds(...). The
		// validateKeepFeasibleAtStart check then catches an infeasible start
		// during dispatch.
		Bounds bnds = new Bounds(
				Matrix.Factory.linkToArray(new double[] {-1.0, -1.0}),
				Matrix.Factory.linkToArray(new double[] {1.0, 1.0}),
				true);

		ToDoubleBiFunction<Matrix, Object> obj = (x, args) ->
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0)
				+ x.getAsDouble(1, 0) * x.getAsDouble(1, 0);
		BiFunction<Matrix, Object, Matrix> grad = (x, args) ->
				Matrix.Factory.linkToArray(new double[] {
						2 * x.getAsDouble(0, 0),
						2 * x.getAsDouble(1, 0)});
		BiFunction<Matrix, Object, Matrix> hess = (x, args) ->
				Matrix.Factory.linkToArray(new double[][] {{2, 0}, {0, 2}});

		// Infeasible start (5, 0) outside [-1, 1] — should throw.
		Matrix x0Bad = Matrix.Factory.linkToArray(new double[] {5.0, 0.0});
		IllegalArgumentException ex = Assertions.assertThrows(
				IllegalArgumentException.class,
				() -> MinimizeTrustConstr.minimizeTrustConstr(
						obj, x0Bad, null,
						grad, hess,
						null, bnds, null,
						1.0e-8, 1.0e-8, 1.0e-8,
						Optional.empty(), null,
						200, 0, null,
						1.0, 1.0, 0.1, 0.1,
						null, false));
		Assertions.assertTrue(ex.getMessage().contains("keep_feasible"),
				"Expected keep_feasible message; got: " + ex.getMessage());
	}
}
