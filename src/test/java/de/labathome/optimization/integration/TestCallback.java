package de.labathome.optimization.integration;

import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.ToDoubleBiFunction;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.LinearConstraint;
import org.scipy.optimize.minimize.MinimizeTrustConstr;
import org.scipy.optimize.minimize.interfaces.IterationCallback;
import org.scipy.optimize.minimize.records.OptimizeResult;
import org.ujmp.core.Matrix;

/**
 * Tests for the {@link IterationCallback} hook plumbed through the full-shape
 * {@link MinimizeTrustConstr#minimizeTrustConstr} entry point.
 *
 * <p>Mirrors scipy's {@code callback} parameter to
 * {@code minimize(method='trust-constr', callback=...)}: invoked once per
 * outer iteration with the running {@code State}, returns {@code true} to
 * request early termination, otherwise {@code false} to continue.
 */
class TestCallback {

	private static final ToDoubleBiFunction<Matrix, Object> rosenFun = (x, args) -> {
		double a = x.getAsDouble(0, 0);
		double b = x.getAsDouble(1, 0);
		return (1.0 - a) * (1.0 - a) + 100.0 * (b - a * a) * (b - a * a);
	};

	private static final BiFunction<Matrix, Object, Matrix> rosenGrad = (x, args) -> {
		double a = x.getAsDouble(0, 0);
		double b = x.getAsDouble(1, 0);
		return Matrix.Factory.linkToArray(new double[] {
				-2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
				200.0 * (b - a * a)});
	};

	private static final BiFunction<Matrix, Object, Matrix> rosenHess = (x, args) -> {
		double a = x.getAsDouble(0, 0);
		double b = x.getAsDouble(1, 0);
		return Matrix.Factory.linkToArray(new double[][] {
				{2.0 - 400.0 * b + 1200.0 * a * a, -400.0 * a},
				{-400.0 * a, 200.0}});
	};

	@Test
	void callbackInvokedOncePerIteration() {
		// Run unconstrained Rosenbrock and count callback invocations. Should
		// equal the iteration count reported by OptimizeResult.
		final int[] callbackInvocations = {0};
		IterationCallback cb = state -> {
			callbackInvocations[0]++;
			return false;  // never request termination
		};

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.5});
		OptimizeResult r = MinimizeTrustConstr.minimizeTrustConstr(
				rosenFun, x0, null,
				rosenGrad, rosenHess,
				null, null, null,
				1.0e-10, 1.0e-10, 1.0e-10,
				Optional.empty(),
				cb,
				1000, 0, null,
				1.0, 1.0, 0.1, 0.1,
				null, false);

		Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 1.0e-5);
		Assertions.assertEquals(r.nIter, callbackInvocations[0],
				"Callback invocations should match nIter; got "
						+ callbackInvocations[0] + " vs " + r.nIter);
	}

	@Test
	void callbackCanRequestEarlyTermination() {
		// Stop after 3 callback invocations.
		final int[] callbackInvocations = {0};
		IterationCallback cb = state -> {
			callbackInvocations[0]++;
			return callbackInvocations[0] >= 3;
		};

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.5});
		OptimizeResult r = MinimizeTrustConstr.minimizeTrustConstr(
				rosenFun, x0, null,
				rosenGrad, rosenHess,
				null, null, null,
				1.0e-10, 1.0e-10, 1.0e-10,
				Optional.empty(),
				cb,
				1000, 0, null,
				1.0, 1.0, 0.1, 0.1,
				null, false);

		Assertions.assertEquals(3, callbackInvocations[0]);
		Assertions.assertEquals(3, r.nIter);
		// Should NOT have converged in 3 iterations on Rosenbrock from (0.5, 1.5)
		// — the callback overrode tolerance-based stopping.
		Assertions.assertTrue(r.fun > 1.0e-3,
				"Callback-stopped run shouldn't have converged; fun=" + r.fun);
	}

	@Test
	void callbackReceivesUpdatedStatePerIteration() {
		// Verify the State passed to the callback carries non-null x and a
		// non-decreasing iteration counter, and that the optimality measure
		// shrinks over iterations on a well-conditioned constrained problem.
		final double[] lastOptimality = {Double.POSITIVE_INFINITY};
		final int[] lastIter = {-1};
		IterationCallback cb = state -> {
			Assertions.assertNotNull(state.x);
			Assertions.assertEquals(2, state.x.getRowCount());
			Assertions.assertTrue(state.nIter > lastIter[0],
					"Iteration counter should be strictly increasing");
			lastIter[0] = state.nIter;
			lastOptimality[0] = state.optimality;
			return false;
		};

		// Quadratic on hyperplane: convex, monotone optimality decrease.
		ToDoubleBiFunction<Matrix, Object> q = (x, args) ->
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0)
				+ x.getAsDouble(1, 0) * x.getAsDouble(1, 0);
		BiFunction<Matrix, Object, Matrix> qg = (x, args) ->
				Matrix.Factory.linkToArray(new double[] {
						2 * x.getAsDouble(0, 0),
						2 * x.getAsDouble(1, 0)});
		BiFunction<Matrix, Object, Matrix> qh = (x, args) ->
				Matrix.Factory.linkToArray(new double[][] {{2, 0}, {0, 2}});

		LinearConstraint eq = new LinearConstraint(
				Matrix.Factory.linkToArray(new double[][] {{1.0, 1.0}}),
				new double[] {2.0}, new double[] {2.0});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {-1.0, 0.5});
		OptimizeResult r = MinimizeTrustConstr.minimizeTrustConstr(
				q, x0, null,
				qg, qh,
				null, null, eq,
				1.0e-10, 1.0e-10, 1.0e-10,
				Optional.empty(),
				cb,
				100, 0, null,
				1.0, 1.0, 0.1, 0.1,
				null, false);

		Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 1.0e-6);
		Assertions.assertEquals(1.0, r.x.getAsDouble(1, 0), 1.0e-6);
		Assertions.assertTrue(lastIter[0] >= 0, "Callback was never invoked");
		Assertions.assertTrue(lastOptimality[0] < 1.0e-6,
				"Final optimality should be tiny; got " + lastOptimality[0]);
	}

	@Test
	void callbackAlsoFiresOnInequalityPath() {
		// Verify the IP path's GlobalStoppingCriteria honors the callback.
		final int[] callbackInvocations = {0};
		IterationCallback cb = state -> {
			callbackInvocations[0]++;
			return false;
		};

		Matrix A = Matrix.Factory.linkToArray(new double[][] {{1.0, 0.0}});
		LinearConstraint ineq = new LinearConstraint(A,
				new double[] {2.0}, new double[] {Double.POSITIVE_INFINITY});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {3.0, 9.0});
		OptimizeResult r = MinimizeTrustConstr.minimizeTrustConstr(
				rosenFun, x0, null,
				rosenGrad, rosenHess,
				null, null, ineq,
				1.0e-8, 1.0e-8, 1.0e-8,
				Optional.empty(),
				cb,
				1000, 0, null,
				1.0, 1.0, 0.1, 0.1,
				null, false);

		Assertions.assertEquals(2.0, r.x.getAsDouble(0, 0), 1.0e-3);
		Assertions.assertTrue(callbackInvocations[0] > 0,
				"Callback should fire at least once on the IP path");
	}
}
