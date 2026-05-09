package de.labathome.optimization;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Optional;
import java.util.function.Function;

import org.junit.jupiter.api.Test;
import de.labathome.trustconstr.CanonicalConstraint;
import de.labathome.trustconstr.LinearConstraint;
import de.labathome.trustconstr.NonlinearConstraint;
import de.labathome.trustconstr.matrix.Matrix;
import de.labathome.trustconstr.records.Bounds;
import de.labathome.trustconstr.records.PreparedConstraint;
import de.labathome.trustconstr.records.Residual;

/**
 * Java translation of subset of scipy {@code optimize/tests/test_constraints.py}
 * relevant to {@link PreparedConstraint}, plus the Java-only Phase 2
 * supporting methods ({@code residual}, {@code violation},
 * {@link CanonicalConstraint#fromPreparedConstraint}).
 */
class TestPreparedConstraint {

	private static final double TOL = 1e-12;

	@Test
	void preparedFromLinearConstraintCachesFAndJ() {
		Matrix A = Matrix.Factory.linkToArray(new double[][] {
				{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
		double[] lb = {-1, -2, -3};
		double[] ub = {1, 2, 3};
		LinearConstraint lc = new LinearConstraint(A, lb, ub, null);
		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 1.0, 1.5});

		PreparedConstraint pc = new PreparedConstraint(lc, x0, Optional.empty());

		// Cached f = A @ x0 = x0
		Matrix f = pc.fun().f();
		RelAbsAssertions.assertRelAbsEquals(0.5, f.getAsDouble(0, 0), TOL);
		RelAbsAssertions.assertRelAbsEquals(1.0, f.getAsDouble(1, 0), TOL);
		RelAbsAssertions.assertRelAbsEquals(1.5, f.getAsDouble(2, 0), TOL);

		// Cached J = A
		Matrix J = pc.fun().J();
		assertEquals(3, J.getRowCount());
		assertEquals(3, J.getColumnCount());

		assertArrayEquals(lb, pc.lb());
		assertArrayEquals(ub, pc.ub());
		assertEquals(3, pc.m());
		assertEquals(3, pc.n());
		assertTrue(pc.source() instanceof LinearConstraint);
	}

	@Test
	void preparedFromNonlinearConstraintCachesFAndJ() {
		Function<Matrix, Matrix> fun = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[] {a * a + b * b, a - b});
		};
		Function<Matrix, Matrix> jac = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			return Matrix.Factory.linkToArray(new double[][] {
					{2 * a, 2 * b}, {1, -1}});
		};
		double[] lb = {0, 0};
		double[] ub = {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY};
		NonlinearConstraint nc = new NonlinearConstraint(fun, jac, lb, ub);

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {1.0, 0.5});
		PreparedConstraint pc = new PreparedConstraint(nc, x0, Optional.empty(), null);

		// Cached f = (a^2 + b^2, a - b) = (1.25, 0.5)
		Matrix f = pc.fun().f();
		RelAbsAssertions.assertRelAbsEquals(1.25, f.getAsDouble(0, 0), TOL);
		RelAbsAssertions.assertRelAbsEquals(0.5, f.getAsDouble(1, 0), TOL);

		// Cached J = ((2, 1), (1, -1))
		Matrix J = pc.fun().J();
		RelAbsAssertions.assertRelAbsEquals(2.0, J.getAsDouble(0, 0), TOL);
		RelAbsAssertions.assertRelAbsEquals(1.0, J.getAsDouble(0, 1), TOL);
		RelAbsAssertions.assertRelAbsEquals(1.0, J.getAsDouble(1, 0), TOL);
		RelAbsAssertions.assertRelAbsEquals(-1.0, J.getAsDouble(1, 1), TOL);

		assertEquals(2, pc.m());
		assertEquals(2, pc.n());
		assertTrue(pc.source() instanceof NonlinearConstraint);
	}

	@Test
	void preparedFromBoundsHasIdentityJacobian() {
		Bounds b = new Bounds(
				Matrix.Factory.linkToArray(new double[] {0, 0, 0}),
				Matrix.Factory.linkToArray(new double[] {1, 1, 1}),
				false);
		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 0.5, 0.5});

		PreparedConstraint pc = new PreparedConstraint(b, x0, Optional.empty());

		// Cached f = x0
		Matrix f = pc.fun().f();
		RelAbsAssertions.assertRelAbsEquals(0.5, f.getAsDouble(0, 0), TOL);
		RelAbsAssertions.assertRelAbsEquals(0.5, f.getAsDouble(1, 0), TOL);
		RelAbsAssertions.assertRelAbsEquals(0.5, f.getAsDouble(2, 0), TOL);

		// Cached J = I
		Matrix J = pc.fun().J();
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				double expected = (i == j) ? 1.0 : 0.0;
				RelAbsAssertions.assertRelAbsEquals(expected, J.getAsDouble(i, j), TOL,
						"J[" + i + "," + j + "]");
			}
		}

		assertEquals(3, pc.m());
		assertEquals(3, pc.n());
		assertTrue(pc.source() instanceof Bounds);
	}

	@Test
	void preparedKeepFeasibleValidationRejectsInfeasibleX0() {
		// Inequality row with kf=true that x0 violates.
		Matrix A = Matrix.Factory.eye(2, 2);
		double[] lb = {0, 0};
		double[] ub = {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY};
		boolean[] kf = {true, false};
		LinearConstraint lc = new LinearConstraint(A, lb, ub, kf);

		Matrix x0Bad = Matrix.Factory.linkToArray(new double[] {-0.5, 0.5});
		assertThrows(IllegalArgumentException.class,
				() -> new PreparedConstraint(lc, x0Bad, Optional.empty()));

		Matrix x0Good = Matrix.Factory.linkToArray(new double[] {0.5, 0.5});
		// Should not throw.
		new PreparedConstraint(lc, x0Good, Optional.empty());
	}

	@Test
	void preparedKeepFeasibleValidationSkipsEqualityRows() {
		// Equality row with kf=true and x0 violates -- scipy excludes equality
		// rows from the kf check (lb==ub branch); see scipy's
		// PreparedConstraint.__init__ "lb != ub" mask.
		Matrix A = Matrix.Factory.eye(2, 2);
		double[] lb = {1, 0};
		double[] ub = {1, Double.POSITIVE_INFINITY};
		boolean[] kf = {true, false};
		LinearConstraint lc = new LinearConstraint(A, lb, ub, kf);

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 0.5}); // violates row 0 eq
		// Should NOT throw -- the kf check skips equality rows.
		new PreparedConstraint(lc, x0, Optional.empty());
	}

	@Test
	void violationOfLinearConstraint() {
		Matrix A = Matrix.Factory.eye(3, 3);
		double[] lb = {0, 0, 0};
		double[] ub = {1, 1, 1};
		LinearConstraint lc = new LinearConstraint(A, lb, ub, null);

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 0.5, 0.5});
		PreparedConstraint pc = new PreparedConstraint(lc, x0, Optional.empty());

		// At feasible x0: violation = 0 everywhere
		Matrix xFeasible = Matrix.Factory.linkToArray(new double[] {0.5, 0.5, 0.5});
		double[] vFeas = pc.violation(xFeasible);
		assertArrayEquals(new double[] {0, 0, 0}, vFeas);

		// At x = (-0.3, 0.5, 1.4): row 0 violates lb (excess 0.3),
		//                          row 1 satisfied,
		//                          row 2 violates ub (excess 0.4)
		Matrix xViol = Matrix.Factory.linkToArray(new double[] {-0.3, 0.5, 1.4});
		double[] vV = pc.violation(xViol);
		RelAbsAssertions.assertRelAbsEquals(0.3, vV[0], TOL);
		RelAbsAssertions.assertRelAbsEquals(0.0, vV[1], TOL);
		RelAbsAssertions.assertRelAbsEquals(0.4, vV[2], TOL);
	}

	@Test
	void linearConstraintResidual() {
		// scipy: LinearConstraint.residual(x) = (A x - lb, ub - A x)
		Matrix A = Matrix.Factory.linkToArray(new double[][] {
				{1, 0}, {0, 1}, {1, 1}});
		double[] lb = {-1, -1, 0};
		double[] ub = {1, 1, 2};
		LinearConstraint lc = new LinearConstraint(A, lb, ub);

		Matrix x = Matrix.Factory.linkToArray(new double[] {0.5, 0.5});
		// A x = [0.5, 0.5, 1.0]
		Residual r = lc.residual(x);
		// sl = A x - lb = [1.5, 1.5, 1.0]
		assertArrayEquals(new double[] {1.5, 1.5, 1.0}, r.sl(), TOL);
		// sb = ub - A x = [0.5, 0.5, 1.0]
		assertArrayEquals(new double[] {0.5, 0.5, 1.0}, r.sb(), TOL);
	}

	@Test
	void boundsResidual() {
		// scipy: Bounds.residual(x) = (x - lb, ub - x)
		Bounds b = new Bounds(
				Matrix.Factory.linkToArray(new double[] {0, -1, 0}),
				Matrix.Factory.linkToArray(new double[] {2, 1, 5}),
				false);
		Matrix x = Matrix.Factory.linkToArray(new double[] {1.0, 0.5, 3.0});

		Residual r = b.residual(x);
		// sl = x - lb = [1, 1.5, 3]
		assertArrayEquals(new double[] {1.0, 1.5, 3.0}, r.sl(), TOL);
		// sb = ub - x = [1, 0.5, 2]
		assertArrayEquals(new double[] {1.0, 0.5, 2.0}, r.sb(), TOL);
	}

	@Test
	void fromPreparedConstraintDispatchesByType() {
		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 0.5});

		// Linear path
		LinearConstraint lc = new LinearConstraint(Matrix.Factory.eye(2, 2),
				new double[] {-1, -1}, new double[] {1, 1});
		PreparedConstraint pcLc = new PreparedConstraint(lc, x0, Optional.empty());
		CanonicalConstraint cc1 = CanonicalConstraint.fromPreparedConstraint(pcLc);
		assertEquals(0, cc1.nEq);
		assertEquals(4, cc1.nIneq);

		// Nonlinear path
		Function<Matrix, Matrix> fun = x -> Matrix.Factory.linkToArray(
				new double[] {x.getAsDouble(0, 0) + x.getAsDouble(1, 0)});
		Function<Matrix, Matrix> jac = x -> Matrix.Factory.linkToArray(
				new double[][] {{1, 1}});
		NonlinearConstraint nc = new NonlinearConstraint(fun, jac,
				new double[] {0}, new double[] {Double.POSITIVE_INFINITY});
		PreparedConstraint pcNc = new PreparedConstraint(nc, x0, Optional.empty(), null);
		CanonicalConstraint cc2 = CanonicalConstraint.fromPreparedConstraint(pcNc);
		assertEquals(0, cc2.nEq);
		assertEquals(1, cc2.nIneq);

		// Bounds path
		Bounds b = new Bounds(
				Matrix.Factory.linkToArray(new double[] {0, 0}),
				Matrix.Factory.linkToArray(new double[] {1, 1}), false);
		PreparedConstraint pcB = new PreparedConstraint(b, x0, Optional.empty());
		CanonicalConstraint cc3 = CanonicalConstraint.fromPreparedConstraint(pcB);
		assertEquals(0, cc3.nEq);
		assertEquals(4, cc3.nIneq);
	}

	@Test
	void initialConstraintsAsCanonicalFromPreparedList() {
		// Replicates the structure of TestCanonicalConstraint.testInitialConstraintsAsCanonical
		// but uses the PreparedConstraint-list overload that mirrors scipy's
		// initial_constraints_as_canonical(n, prepared_constraints, sparse_jacobian).
		final int n = 3;
		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 0.5, 0.5});

		LinearConstraint lc = new LinearConstraint(Matrix.Factory.eye(3, 3),
				new double[] {-1, 0, -1}, new double[] {1, Double.POSITIVE_INFINITY, 1});
		Function<Matrix, Matrix> fun = x -> Matrix.Factory.linkToArray(
				new double[] {x.getAsDouble(0, 0) * x.getAsDouble(0, 0)
						+ x.getAsDouble(1, 0) * x.getAsDouble(1, 0)});
		Function<Matrix, Matrix> jac = x -> Matrix.Factory.linkToArray(new double[][] {
				{2 * x.getAsDouble(0, 0), 2 * x.getAsDouble(1, 0), 0}});
		NonlinearConstraint nc = new NonlinearConstraint(fun, jac,
				new double[] {0.1}, new double[] {Double.POSITIVE_INFINITY});

		for (boolean sparseJacobian : new boolean[] {false, true}) {
			PreparedConstraint pcLc = new PreparedConstraint(lc, x0, Optional.empty());
			PreparedConstraint pcNc = new PreparedConstraint(nc, x0, Optional.empty(), null);

			CanonicalConstraint.InitialCanonical r =
					CanonicalConstraint.initialConstraintsAsCanonical(
							n, List.of(pcLc, pcNc), sparseJacobian);

			// Source 1 (linear): no equality rows, nIneq:
			//   row 0 interval (lb=-1, ub=1) -> 2 rows (upper, lower)
			//   row 1 greater (lb=0, ub=inf) -> 1 row
			//   row 2 interval (lb=-1, ub=1) -> 2 rows
			// Total: 5 ineq from source 1.
			// Source 2 (nonlinear): no equality rows, nIneq = 1 (greater).
			// Total: 0 eq, 6 ineq.
			assertEquals(0, r.cEq().getRowCount());
			assertEquals(6, r.cIneq().getRowCount());
			assertEquals(0, r.JEq().getRowCount());
			assertEquals(6, r.JIneq().getRowCount());
			assertEquals(n, r.JIneq().getColumnCount());
		}
	}
}
