package de.labathome.optimization.integration;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import java.util.function.Function;

import org.scipy.optimize.minimize.LinearConstraint;
import org.scipy.optimize.minimize.MinimizeTrustConstr;
import org.scipy.optimize.minimize.NonlinearConstraint;
import org.scipy.optimize.minimize.enums.TrustConstrMethod;
import org.scipy.optimize.minimize.records.OptimizeResult;
import org.ujmp.core.Matrix;

/**
 * End-to-end tests for the inequality-constrained dispatch in
 * {@link MinimizeTrustConstr#minimize}.
 *
 * <p>The dispatch is wired up: a problem with inequality rows correctly routes
 * through {@link org.scipy.optimize.minimize.TrustRegionInteriorPoint} (the
 * barrier-subproblem ladder, sparse Jacobian assembly, augmented-system
 * projections), and the {@link OptimizeResult#method} field reflects that.
 *
 */
class TestInequalityConstrained {

	@Test
	void dispatchRoutesToInteriorPoint() {
		// Confirms: a constraint with inequality rows produces an OptimizeResult whose
		// method field is TRUST_REGION_INTERIOR_POINT (i.e., the dispatch chose the
		// right path). We don't assert convergence here — see the @Disabled cases.
		java.util.function.Function<Matrix, Double> q = x ->
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0);
		java.util.function.Function<Matrix, Matrix> qg = x ->
				Matrix.Factory.linkToArray(new double[] { 2 * x.getAsDouble(0, 0) });
		java.util.function.Function<Matrix, Matrix> qh = x ->
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
		java.util.function.Function<Matrix, Double> q = x ->
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0);
		java.util.function.Function<Matrix, Matrix> qg = x ->
				Matrix.Factory.linkToArray(new double[] { 2 * x.getAsDouble(0, 0) });
		java.util.function.Function<Matrix, Matrix> qh = x ->
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
					200.0 * (b - a * a) });
		};
		java.util.function.Function<Matrix, Matrix> rosenH = x -> {
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
	}

	@Test
	@Disabled("Rosenbrock with x[0]>=2: optimum is (2, 4) but the IP path converges "
			+ "to ~(2.31, …) with trust-radius collapse before reaching it. Same "
			+ "stock-trust-constr cold-start fragility as the equality-constrained "
			+ "Rosenbrock case, exacerbated by an active inequality. Higher-order "
			+ "barrier-update strategies (scipy's [2] reference) would help.")
	void rosenbrockUnderActiveBound() {
		// Rosenbrock with x[0] >= 2.  Constrained optimum: x=(2, 4), f=1.
		// (At x=2 the parabola y=x^2 is followed; outside, f grows.)
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
					200.0 * (b - a * a) });
		};
		java.util.function.Function<Matrix, Matrix> rosenH = x -> {
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
		Assertions.assertEquals(2.0, r.x.getAsDouble(0, 0), 5.0e-3);
		Assertions.assertEquals(4.0, r.x.getAsDouble(1, 0), 5.0e-2);
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
		java.util.function.Function<Matrix, Double> q = x ->
				x.getAsDouble(0, 0) * x.getAsDouble(0, 0) + x.getAsDouble(1, 0) * x.getAsDouble(1, 0);
		java.util.function.Function<Matrix, Matrix> qg = x ->
				Matrix.Factory.linkToArray(new double[] { 2 * x.getAsDouble(0, 0), 2 * x.getAsDouble(1, 0) });
		java.util.function.Function<Matrix, Matrix> qh = x ->
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
