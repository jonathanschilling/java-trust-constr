package de.labathome.optimization;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.matrix.MatrixOps;
import org.scipy.optimize.minimize.NonlinearConstraint;
import org.scipy.optimize.minimize.matrix.Matrix;


class TestNonlinearConstraint {

	private static final double TOL = 1.0e-12;
	private static final double NEG_INF = Double.NEGATIVE_INFINITY;
	private static final double POS_INF = Double.POSITIVE_INFINITY;

	@Test
	void testEqualityScalar() {
		// fun(x) = x[0]^2 + x[1]^2; constraint: fun(x) == 1 (unit circle)
		java.util.function.Function<Matrix, Matrix> fun = x -> {
			double x0 = x.getAsDouble(0, 0);
			double x1 = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[] { x0 * x0 + x1 * x1 });
		};
		java.util.function.Function<Matrix, Matrix> jac = x -> {
			double x0 = x.getAsDouble(0, 0);
			double x1 = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[][] { { 2 * x0, 2 * x1 } });
		};
		NonlinearConstraint c = new NonlinearConstraint(fun, jac, new double[] {1.0}, new double[] {1.0});
		Assertions.assertEquals(1, c.nEq());
		Assertions.assertEquals(0, c.nIneq());

		Matrix x = Matrix.Factory.linkToArray(new double[] {0.6, 0.8});
		// f(x) = 0.36 + 0.64 = 1; constrEq = 1 - 1 = 0
		RelAbsAssertions.assertArrayRelAbsEquals(new double[] {0.0}, c.constrEq(x).toColumnArray(), TOL);
		// jacEq = [2*0.6, 2*0.8] = [1.2, 1.6]
		RelAbsAssertions.assertArrayRelAbsEquals(new double[][] { {1.2, 1.6} }, c.jacEq(x).toDoubleArray(), TOL);
	}

	@Test
	void testInequalityUpper() {
		// fun(x) = x[0]; constraint: fun(x) <= 5
		java.util.function.Function<Matrix, Matrix> fun = x ->
				Matrix.Factory.linkToArray(new double[] { x.getAsDouble(0, 0) });
		java.util.function.Function<Matrix, Matrix> jac = x ->
				Matrix.Factory.linkToArray(new double[][] { { 1.0 } });
		NonlinearConstraint c = new NonlinearConstraint(fun, jac, new double[] {NEG_INF}, new double[] {5.0});
		Assertions.assertEquals(0, c.nEq());
		Assertions.assertEquals(1, c.nIneq());

		Matrix x = Matrix.Factory.linkToArray(new double[] {7.0});
		// constrIneq = +1*(7 - 5) = 2  (positive => violated)
		RelAbsAssertions.assertArrayRelAbsEquals(new double[] {2.0}, c.constrIneq(x).toColumnArray(), TOL);
	}

	@Test
	void testFiveRowMixedBounds() {
		// Mirrors scipy's _trustregion_constr/tests/test_canonical_constraint.py::test_nonlinear_constraint:
		//   lb = [-10, 3,  -inf, -inf, -5]
		//   ub = [ 10, 3,   inf,    3, inf]
		// Row classification:
		//   row 0: [-10, 10] interval -> 2 ineq rows (upper, lower)
		//   row 1: [3, 3] equality    -> 1 eq row
		//   row 2: [-inf, inf]        -> dropped
		//   row 3: [-inf, 3] upper    -> 1 ineq row
		//   row 4: [-5, inf] lower    -> 1 ineq row
		// So nEq = 1, nIneq = 4 — same as scipy's CanonicalConstraint.
		java.util.function.Function<Matrix, Matrix> fun = x -> {
			double a = x.getAsDouble(0, 0);
			return Matrix.Factory.linkToArray(new double[] {
					a, a, a, a, a});
		};
		java.util.function.Function<Matrix, Matrix> jac = x ->
				Matrix.Factory.linkToArray(new double[][] {
						{1.0}, {1.0}, {1.0}, {1.0}, {1.0}});

		NonlinearConstraint c = new NonlinearConstraint(fun, jac,
				new double[] {-10, 3, NEG_INF, NEG_INF, -5},
				new double[] { 10, 3, POS_INF, 3,      POS_INF});
		Assertions.assertEquals(1, c.nEq());
		Assertions.assertEquals(4, c.nIneq());

		// At x0 = (4,), fx = (4, 4, 4, 4, 4).
		Matrix x0 = Matrix.Factory.linkToArray(new double[] {4.0});

		// constrEq = (fx[1] - lb[1],) = (4 - 3,) = (1,)
		RelAbsAssertions.assertArrayRelAbsEquals(new double[] {1.0},
				c.constrEq(x0).toColumnArray(), TOL);

		// constrIneq order (scipy 4-block: less, greater, interval-upper, interval-lower):
		//   row 3 upper (less):     +1 * (4 - 3) = 1
		//   row 4 lower (greater):  -1 * (4 - (-5)) = -9
		//   row 0 upper (interval): +1 * (4 - 10) = -6
		//   row 0 lower (interval): -1 * (4 - (-10)) = -14
		RelAbsAssertions.assertArrayRelAbsEquals(new double[] {1.0, -9.0, -6.0, -14.0},
				c.constrIneq(x0).toColumnArray(), TOL);
	}

	@Test
	void testTwoSidedSplitsIntoUpperAndLowerRows() {
		// fun(x) = x[0]; bound: 0 <= x[0] <= 1
		java.util.function.Function<Matrix, Matrix> fun = x ->
				Matrix.Factory.linkToArray(new double[] { x.getAsDouble(0, 0) });
		java.util.function.Function<Matrix, Matrix> jac = x ->
				Matrix.Factory.linkToArray(new double[][] { { 1.0 } });
		NonlinearConstraint c = new NonlinearConstraint(fun, jac,
				new double[] {0.0}, new double[] {1.0});
		Assertions.assertEquals(0, c.nEq());
		Assertions.assertEquals(2, c.nIneq());

		Matrix x = Matrix.Factory.linkToArray(new double[] {0.5});
		// constrIneq = [+1*(0.5 - 1), -1*(0.5 - 0)] = [-0.5, -0.5]
		RelAbsAssertions.assertArrayRelAbsEquals(new double[] {-0.5, -0.5},
				c.constrIneq(x).toColumnArray(), TOL);
	}

	@Test
	void testLagrangianContributionPacksMultipliersBackToOriginalRows() {
		// Mirrors the Hessian assertion in scipy
		// _trustregion_constr/tests/test_canonical_constraint.py::test_nonlinear_constraint:
		// the canonical (vEq, vIneq) multipliers must be packed back to the
		// original m-vector of row multipliers — eq rows get vEq directly,
		// upper-bound ineq rows get +vIneq, lower-bound ineq rows get -vIneq —
		// before being passed to the user's hess(x, v).
		//
		// Five rows, lb = [-10, 3, -inf, -inf, -5], ub = [10, 3, inf, 3, inf]:
		//   row 0: interval -> 2 ineq
		//   row 1: equality -> 1 eq
		//   row 2: free (dropped)
		//   row 3: upper-only (less) -> 1 ineq
		//   row 4: lower-only (greater) -> 1 ineq
		// vIneq order (scipy 4-block): [row3 less, row4 greater, row0 ub, row0 lb].
		// hess(x, v) gets the row-multiplier-packed v of length m=5.
		java.util.function.Function<Matrix, Matrix> fun = x -> {
			double a = x.getAsDouble(0, 0);
			return Matrix.Factory.linkToArray(new double[] {a, a, a, a, a});
		};
		java.util.function.Function<Matrix, Matrix> jac = x ->
				Matrix.Factory.linkToArray(new double[][] {
						{1.0}, {1.0}, {1.0}, {1.0}, {1.0}});
		// Expose only rows 0 and 3 with curvature; others contribute 0.
		// Total = 2*v[0] + 4*v[3] at the scalar position [0,0].
		java.util.function.BiFunction<Matrix, Matrix, Matrix> hess = (x, v) -> {
			double total = 2.0 * v.getAsDouble(0, 0) + 4.0 * v.getAsDouble(3, 0);
			return Matrix.Factory.linkToArray(new double[][] {{total}});
		};

		NonlinearConstraint c = new NonlinearConstraint(fun, jac, hess,
				new double[] {-10, 3, NEG_INF, NEG_INF, -5},
				new double[] { 10, 3, POS_INF, 3, POS_INF},
				null);

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {1.0});

		// vEq[0] -> v[1]; ignored by our hess (row 1 has no curvature).
		// vIneq order (scipy 4-block): [row3_ub (less), row4_lb (greater), row0_ub, row0_lb].
		//   v[3] += +1*vIneq[0] = 0.3
		//   v[4] += (-1)*vIneq[1] = -0.1 (no curvature contribution)
		//   v[0] += +1*vIneq[2] + (-1)*vIneq[3] = 0.7 - 0.2 = 0.5
		// hess total = 2*0.5 + 4*0.3 = 2.2.
		double[] vEq = {0.5};
		double[] vIneq = {0.3, 0.1, 0.7, 0.2};
		Matrix H = c.lagrangianContribution(x0, vEq, vIneq);
		Assertions.assertNotNull(H);
		RelAbsAssertions.assertRelAbsEquals(2.2, H.getAsDouble(0, 0), TOL);
	}
}
