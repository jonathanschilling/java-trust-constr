package de.labathome.optimization.integration;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import java.util.function.Function;

import de.labathome.trustconstr.LinearConstraint;
import de.labathome.trustconstr.MinimizeTrustConstr;
import de.labathome.trustconstr.NonlinearConstraint;
import de.labathome.trustconstr.enums.TrustConstrMethod;
import de.labathome.trustconstr.records.Bounds;
import de.labathome.trustconstr.records.OptimizeResult;
import de.labathome.trustconstr.matrix.Matrix;

/**
 * End-to-end tests for the inequality-constrained dispatch in
 * {@link MinimizeTrustConstr#minimize}.
 *
 * <p>The dispatch is wired up: a problem with inequality rows correctly routes
 * through {@link de.labathome.trustconstr.TrustRegionInteriorPoint} (the
 * barrier-subproblem ladder, sparse Jacobian assembly, augmented-system
 * projections), and the {@link OptimizeResult#method} field reflects that.
 *
 */
class TestInequalityConstrained {

	@Test
	void dispatchRoutesToInteriorPoint() {
		// Confirms: a constraint with inequality rows produces an OptimizeResult whose
		// method field is TRUST_REGION_INTERIOR_POINT (i.e., the dispatch chose the
		// right path). We don't assert convergence here -- see the @Disabled cases.
		Function<Matrix, Double> q = x ->
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0);
		Function<Matrix, Matrix> qg = x ->
				Matrix.Factory.linkToArray(new double[] { 2 * x.getAsDouble(0, 0) });
		Function<Matrix, Matrix> qh = x ->
				Matrix.Factory.linkToArray(new double[][] { {2} });
		Matrix A = Matrix.Factory.linkToArray(new double[][] { { 1.0 } });
		LinearConstraint ineq = new LinearConstraint(A,
				new double[] {1.0}, new double[] {Double.POSITIVE_INFINITY});
		Matrix x0 = Matrix.Factory.linkToArray(new double[] {1.5});

		OptimizeResult r = MinimizeTrustConstr.minimize(q, qg, qh, x0, ineq,
				50, 1.0e-8, 1.0e-8);

		Assertions.assertEquals(TrustConstrMethod.TRUST_REGION_INTERIOR_POINT, r.method);
		Assertions.assertNotNull(r.x);
		Assertions.assertEquals(1, r.x.getRowCount());
	}

	@Test
	void scalarQuadraticUnderActiveLowerBound() {
		Function<Matrix, Double> q = x ->
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0);
		Function<Matrix, Matrix> qg = x ->
				Matrix.Factory.linkToArray(new double[] { 2 * x.getAsDouble(0, 0) });
		Function<Matrix, Matrix> qh = x ->
				Matrix.Factory.linkToArray(new double[][] { {2} });
		Matrix A = Matrix.Factory.linkToArray(new double[][] { { 1.0 } });
		LinearConstraint ineq = new LinearConstraint(A,
				new double[] {1.0}, new double[] {Double.POSITIVE_INFINITY});
		Matrix x0 = Matrix.Factory.linkToArray(new double[] {1.5});
		OptimizeResult r = MinimizeTrustConstr.minimize(q, qg, qh, x0, ineq,
				1000, 1.0e-8, 1.0e-8);
		Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 1e-3);
	}

	@Test
	void rosenbrockUnderInactiveBound() {
		// Rosenbrock with x[0] >= -2: bound is inactive at the optimum (1, 1).
		// Tests that the IP path doesn't wander when the constraint is loose.
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
		Function<Matrix, Matrix> rosenH = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[][] {
					{ 2.0 - 400.0 * b + 1200.0 * a * a, -400.0 * a },
					{ -400.0 * a, 200.0 } });
		};

		Matrix A = Matrix.Factory.linkToArray(new double[][] { { 1.0, 0.0 } });
		LinearConstraint ineq = new LinearConstraint(A,
				new double[] {-2.0}, new double[] {Double.POSITIVE_INFINITY});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.5});
		OptimizeResult r = MinimizeTrustConstr.minimize(rosen, rosenG, rosenH, x0, ineq,
				1000, 1.0e-8, 1.0e-8);
		Assertions.assertEquals(TrustConstrMethod.TRUST_REGION_INTERIOR_POINT, r.method);
		Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 5.0e-3);
		Assertions.assertEquals(1.0, r.x.getAsDouble(1, 0), 5.0e-3);

		// Convergence signal -- the SQP-loop's Lagrangian gradient norm.
		Assertions.assertTrue(r.optimality < 1.0e-6,
				"Expected r.optimality < 1e-6, got " + r.optimality);
		// Eval counters: must be positive and reasonable.
		Assertions.assertTrue(r.numFunctionEval > 0 && r.numFunctionEval < 1000,
				"Unexpected fun-eval count: " + r.numFunctionEval);
		Assertions.assertTrue(r.numJacobianEval > 0 && r.numJacobianEval < 1000,
				"Unexpected grad-eval count: " + r.numJacobianEval);
		Assertions.assertTrue(r.numHessianEval > 0 && r.numHessianEval < 1000,
				"Unexpected hess-eval count: " + r.numHessianEval);
		// Lagrangian gradient: g + Jineq^T lambda. The bound is inactive (lambda ~= 0),
		// so this reduces to the ordinary gradient at Rosenbrock's optimum,
		// which is zero.
		Assertions.assertNotNull(r.lagrangianGrad);
		Assertions.assertTrue(r.lagrangianGrad.normInf() < 1.0e-4,
				"Lagrangian gradient too large: " + r.lagrangianGrad.normInf());
		// IP-path-only fields: barrierParameter / barrierTolerance should
		// have decayed below their starting values (0.1 each) by termination.
		Assertions.assertTrue(r.barrierParameter > 0 && r.barrierParameter < 0.1,
				"Expected barrierParameter to have decayed below 0.1; got "
						+ r.barrierParameter);
		Assertions.assertTrue(r.barrierTolerance > 0 && r.barrierTolerance < 0.1,
				"Expected barrierTolerance to have decayed below 0.1; got "
						+ r.barrierTolerance);
	}

	@Test
	void rosenbrockUnderActiveBound() {
		// Rosenbrock with x[0] >= 2.  Constrained optimum: x=(2, 4), f=1.
		// (At x=2 the parabola y=x^2 is followed; outside, f grows.)
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
		Function<Matrix, Matrix> rosenH = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[][] {
					{ 2.0 - 400.0 * b + 1200.0 * a * a, -400.0 * a },
					{ -400.0 * a, 200.0 } });
		};

		Matrix A = Matrix.Factory.linkToArray(new double[][] { { 1.0, 0.0 } });
		LinearConstraint ineq = new LinearConstraint(A,
				new double[] {2.0}, new double[] {Double.POSITIVE_INFINITY});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {3.0, 9.0});
		OptimizeResult r = MinimizeTrustConstr.minimize(rosen, rosenG, rosenH, x0, ineq,
				1000, 1.0e-8, 1.0e-8);
		Assertions.assertEquals(TrustConstrMethod.TRUST_REGION_INTERIOR_POINT, r.method);
		Assertions.assertEquals(2.0, r.x.getAsDouble(0, 0), 1.0e-3);
		Assertions.assertEquals(4.0, r.x.getAsDouble(1, 0), 1.0e-2);
		// KKT at the optimum: g + Jineq^T lambda = 0. The bound is active (lambda != 0)
		// so this is a sharper test than the inactive case -- only correct
		// multiplier recovery makes the Lagrangian gradient vanish.
		Assertions.assertNotNull(r.lagrangianGrad);
		Assertions.assertTrue(r.lagrangianGrad.normInf() < 1.0e-4,
				"Lagrangian gradient too large at active bound: " + r.lagrangianGrad.normInf());
	}

	@Test
	void quadraticUnderVariableBounds() {
		// minimize (x - 5)^2 + (y - 5)^2  s.t.  0 <= x <= 1, 0 <= y <= 2
		// Optimum at the upper-right corner: (1, 2), f = 16 + 9 = 25.
		Function<Matrix, Double> q = x -> {
			double dx = x.getAsDouble(0, 0) - 5.0;
			double dy = x.getAsDouble(1, 0) - 5.0;
			return dx * dx + dy * dy;
		};
		Function<Matrix, Matrix> qg = x ->
				Matrix.Factory.linkToArray(new double[] {
						2 * (x.getAsDouble(0, 0) - 5.0),
						2 * (x.getAsDouble(1, 0) - 5.0) });
		Function<Matrix, Matrix> qh = x ->
				Matrix.Factory.linkToArray(new double[][] { {2, 0}, {0, 2} });

		Bounds b = new Bounds(
				Matrix.Factory.linkToArray(new double[] {0.0, 0.0}),
				Matrix.Factory.linkToArray(new double[] {1.0, 2.0}),
				false);
		LinearConstraint asConstraint = LinearConstraint.fromBounds(b);

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.0});
		OptimizeResult r = MinimizeTrustConstr.minimize(q, qg, qh, x0, asConstraint,
				1000, 1.0e-8, 1.0e-8);
		Assertions.assertEquals(TrustConstrMethod.TRUST_REGION_INTERIOR_POINT, r.method);
		Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 5.0e-3);
		Assertions.assertEquals(2.0, r.x.getAsDouble(1, 0), 5.0e-3);
	}

	@Test
	void quadraticInsideDisk() {
		// minimize (x-2)^2 + (y-2)^2  s.t.  x^2 + y^2 <= 1  (interior of unit disk)
		// Optimum lies on the disk boundary closest to (2, 2): x* = (1/sqrt(2), 1/sqrt(2)).
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

		Function<Matrix, Matrix> cFun = x -> Matrix.Factory.linkToArray(new double[] {
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0) + x.getAsDouble(1, 0) * x.getAsDouble(1, 0)
		});
		Function<Matrix, Matrix> cJac = x -> Matrix.Factory.linkToArray(new double[][] {
				{ 2 * x.getAsDouble(0, 0), 2 * x.getAsDouble(1, 0) }
		});
		// Use an upper-only nonlinear inequality (x^2 + y^2 <= 1).
		NonlinearConstraint c = new NonlinearConstraint(cFun, cJac,
				new double[] {Double.NEGATIVE_INFINITY}, new double[] {1.0});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 0.5});
		OptimizeResult r = MinimizeTrustConstr.minimize(q, qg, qh, x0, c,
				1000, 1.0e-8, 1.0e-8);

		Assertions.assertEquals(TrustConstrMethod.TRUST_REGION_INTERIOR_POINT, r.method);
		double expected = 1.0 / Math.sqrt(2.0);
		Assertions.assertEquals(expected, r.x.getAsDouble(0, 0), 5.0e-3);
		Assertions.assertEquals(expected, r.x.getAsDouble(1, 0), 5.0e-3);
	}

	@Test
	void quadraticUnderActiveLowerBound() {
		Function<Matrix, Double> q = x ->
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0) + x.getAsDouble(1, 0) * x.getAsDouble(1, 0);
		Function<Matrix, Matrix> qg = x ->
				Matrix.Factory.linkToArray(new double[] { 2 * x.getAsDouble(0, 0), 2 * x.getAsDouble(1, 0) });
		Function<Matrix, Matrix> qh = x ->
				Matrix.Factory.linkToArray(new double[][] { {2, 0}, {0, 2} });

		Matrix A = Matrix.Factory.linkToArray(new double[][] { { 1.0, 0.0 } });
		LinearConstraint ineq = new LinearConstraint(A,
				new double[] {1.0}, new double[] {Double.POSITIVE_INFINITY});

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {2.0, 2.0});
		OptimizeResult r = MinimizeTrustConstr.minimize(q, qg, qh, x0, ineq,
				1000, 1.0e-8, 1.0e-8);
		Assertions.assertEquals(1.0, r.x.getAsDouble(0, 0), 5e-3);
		Assertions.assertEquals(0.0, r.x.getAsDouble(1, 0), 5e-3);
	}
}
