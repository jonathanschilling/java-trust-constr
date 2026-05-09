package de.labathome.optimization;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.CanonicalConstraint;
import org.scipy.optimize.minimize.LinearConstraint;
import org.scipy.optimize.minimize.NonlinearConstraint;
import org.scipy.optimize.minimize.matrix.Matrix;
import org.scipy.optimize.minimize.records.EqIneqSplit;

/**
 * Java translation of scipy
 * {@code _trustregion_constr/tests/test_canonical_constraint.py}. Each
 * Java {@code @Test} method mirrors one of the six scipy test functions:
 * {@code test_bounds_cases}, {@code test_nonlinear_constraint},
 * {@code test_concatenation}, {@code test_empty},
 * {@code test_initial_constraints_as_canonical}, and
 * {@code test_initial_constraints_as_canonical_empty}.
 *
 * <p>Where scipy uses {@code Bounds(...)} with per-row {@code keep_feasible},
 * the Java equivalent is a {@link LinearConstraint} with identity {@code A},
 * because the Java {@code Bounds} record only carries a scalar
 * {@code keepFeasible}. This is a syntactic divergence; the canonical-form
 * residuals, Jacobians, and {@code keepFeasible} array are identical.
 */
class TestCanonicalConstraint {

	private static final double TOL = 1e-12;

	private static final double[] INF_LB(int n) {
		double[] a = new double[n];
		Arrays.fill(a, Double.NEGATIVE_INFINITY);
		return a;
	}

	private static final double[] INF_UB(int n) {
		double[] a = new double[n];
		Arrays.fill(a, Double.POSITIVE_INFINITY);
		return a;
	}

	@Test
	void testBoundsCases() {
		// Sub-case 1: no constraints (lb = -inf, ub = +inf for all rows).
		{
			LinearConstraint lc = new LinearConstraint(
					Matrix.Factory.eye(2, 2), INF_LB(2), INF_UB(2), null);
			CanonicalConstraint c = CanonicalConstraint.fromLinearConstraint(lc);
			assertEquals(0, c.nEq);
			assertEquals(0, c.nIneq);

			Matrix x0 = Matrix.Factory.linkToArray(new double[] {-1.0, 2.0});
			EqIneqSplit f = c.fun(x0);
			EqIneqSplit J = c.jac(x0);
			assertEquals(0, f.eq().getRowCount());
			assertEquals(0, f.ineq().getRowCount());
			assertEquals(0, J.eq().getRowCount());
			assertEquals(2, J.eq().getColumnCount());
			assertEquals(0, J.ineq().getRowCount());
			assertEquals(2, J.ineq().getColumnCount());

			assertArrayEquals(new boolean[0], c.keepFeasible);
		}

		// Sub-case 2: infinite lower bound, mixed finite/infinite upper bounds,
		// per-row keep_feasible.
		// scipy: Bounds(-inf, [0, inf, 1], [False, True, True]); Java equivalent
		// is the linear constraint with identity A and per-row keep_feasible.
		{
			Matrix A = Matrix.Factory.eye(3, 3);
			double[] lb = INF_LB(3);
			double[] ub = {0.0, Double.POSITIVE_INFINITY, 1.0};
			boolean[] kf = {false, true, true};
			LinearConstraint lc = new LinearConstraint(A, lb, ub, kf);
			CanonicalConstraint c = CanonicalConstraint.fromLinearConstraint(lc);

			assertEquals(0, c.nEq);
			assertEquals(2, c.nIneq);

			Matrix x0 = Matrix.Factory.linkToArray(new double[] {-1.0, -2.0, -3.0});
			EqIneqSplit f = c.fun(x0);
			EqIneqSplit J = c.jac(x0);

			// Expected ineq residuals: [x[0] - 0, x[2] - 1] = [-1, -4]
			assertEquals(0, f.eq().getRowCount());
			assertEquals(2, f.ineq().getRowCount());
			RelAbsAssertions.assertRelAbsEquals(-1.0, f.ineq().getAsDouble(0, 0), TOL);
			RelAbsAssertions.assertRelAbsEquals(-4.0, f.ineq().getAsDouble(1, 0), TOL);

			// Expected ineq Jacobian: [[1, 0, 0], [0, 0, 1]]
			double[][] expectedJIneq = {{1, 0, 0}, {0, 0, 1}};
			assertMatrixEquals(expectedJIneq, J.ineq());

			// keep_feasible follows row order in the canonical 4-block ineq
			// listing: less rows (rows 0 and 2 of the original A) -> kf flags
			// from those rows.
			assertArrayEquals(new boolean[] {false, true}, c.keepFeasible);
		}

		// Sub-case 3: infinite upper bound, mixed finite/infinite lower bounds,
		// per-row keep_feasible.
		// scipy: Bounds([0, 1, -inf], inf, [True, False, True]); Java equivalent
		// uses identity-A LinearConstraint.
		{
			Matrix A = Matrix.Factory.eye(3, 3);
			double[] lb = {0.0, 1.0, Double.NEGATIVE_INFINITY};
			double[] ub = INF_UB(3);
			boolean[] kf = {true, false, true};
			LinearConstraint lc = new LinearConstraint(A, lb, ub, kf);
			CanonicalConstraint c = CanonicalConstraint.fromLinearConstraint(lc);

			assertEquals(0, c.nEq);
			assertEquals(2, c.nIneq);

			Matrix x0 = Matrix.Factory.linkToArray(new double[] {1.0, 2.0, 3.0});
			EqIneqSplit f = c.fun(x0);
			EqIneqSplit J = c.jac(x0);

			// Expected ineq residuals: [0 - x[0], 1 - x[1]] = [-1, -1]
			RelAbsAssertions.assertRelAbsEquals(-1.0, f.ineq().getAsDouble(0, 0), TOL);
			RelAbsAssertions.assertRelAbsEquals(-1.0, f.ineq().getAsDouble(1, 0), TOL);

			// Expected ineq Jacobian: [[-1, 0, 0], [0, -1, 0]]
			double[][] expectedJIneq = {{-1, 0, 0}, {0, -1, 0}};
			assertMatrixEquals(expectedJIneq, J.ineq());

			assertArrayEquals(new boolean[] {true, false}, c.keepFeasible);
		}

		// Sub-case 4: interval constraint with mixed shapes per row.
		// scipy: Bounds([-1, -inf, 2, 3], [1, inf, 10, 3], [False, True, True, True])
		{
			Matrix A = Matrix.Factory.eye(4, 4);
			double[] lb = {-1.0, Double.NEGATIVE_INFINITY, 2.0, 3.0};
			double[] ub = {1.0, Double.POSITIVE_INFINITY, 10.0, 3.0};
			boolean[] kf = {false, true, true, true};
			LinearConstraint lc = new LinearConstraint(A, lb, ub, kf);
			CanonicalConstraint c = CanonicalConstraint.fromLinearConstraint(lc);

			assertEquals(1, c.nEq);   // row 3: lb == ub == 3
			assertEquals(4, c.nIneq); // row 0 interval (2 rows), row 2 interval (2 rows)

			Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.0, 10.0, 8.0, 5.0});
			EqIneqSplit f = c.fun(x0);
			EqIneqSplit J = c.jac(x0);

			// Equality: x[3] - 3 = 5 - 3 = 2
			RelAbsAssertions.assertRelAbsEquals(2.0, f.eq().getAsDouble(0, 0), TOL);

			// Ineq order: less | greater | interval-upper | interval-lower.
			// Row 1 was less but has ub=inf; that's a no-constraint row (both
			// bounds infinite -- wait, row 1 is lb=-inf, ub=+inf, so it's
			// dropped). Re-check rows:
			//   row 0: lb=-1, ub=1     -> interval (both finite, lb<ub)
			//   row 1: lb=-inf, ub=inf -> dropped
			//   row 2: lb=2, ub=10     -> interval (both finite, lb<ub)
			//   row 3: lb=3, ub=3      -> equality
			// So less=[], greater=[], interval-upper=[0, 2], interval-lower=[0, 2].
			// Expected ineq:
			//   row 0 upper: x[0] - 1 = -1
			//   row 2 upper: x[2] - 10 = -2
			//   row 0 lower: -1 - x[0] = -1
			//   row 2 lower: 2 - x[2] = -6
			RelAbsAssertions.assertRelAbsEquals(-1.0, f.ineq().getAsDouble(0, 0), TOL);
			RelAbsAssertions.assertRelAbsEquals(-2.0, f.ineq().getAsDouble(1, 0), TOL);
			RelAbsAssertions.assertRelAbsEquals(-1.0, f.ineq().getAsDouble(2, 0), TOL);
			RelAbsAssertions.assertRelAbsEquals(-6.0, f.ineq().getAsDouble(3, 0), TOL);

			double[][] expectedJEq = {{0, 0, 0, 1}};
			assertMatrixEquals(expectedJEq, J.eq());

			double[][] expectedJIneq = {
					{1, 0, 0, 0}, {0, 0, 1, 0}, {-1, 0, 0, 0}, {0, 0, -1, 0}};
			assertMatrixEquals(expectedJIneq, J.ineq());

			// kf in canonical-row order:
			// interval-upper rows 0, 2 -> kf[0], kf[2] = false, true
			// interval-lower rows 0, 2 -> kf[0], kf[2] = false, true
			assertArrayEquals(new boolean[] {false, true, false, true}, c.keepFeasible);
		}
	}

	@Test
	void testNonlinearConstraint() {
		final int n = 3;
		final int m = 5;

		QuadraticConstraint q = QuadraticConstraint.deterministic(n, m);
		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.1, 0.2, 0.3});
		Matrix fx0 = q.fun(x0);
		Matrix Jx0 = q.jac(x0);
		double[] f = new double[m];
		for (int i = 0; i < m; ++i) f[i] = fx0.getAsDouble(i, 0);

		double[] lb = {-10, 3, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, -5};
		double[] ub = {10, 3, Double.POSITIVE_INFINITY, 3, Double.POSITIVE_INFINITY};
		boolean[] kf = {true, false, false, true, false};

		Function<Matrix, Matrix> funFn = q::fun;
		Function<Matrix, Matrix> jacFn = q::jac;
		BiFunction<Matrix, Matrix, Matrix> hessFn = q::hess;
		NonlinearConstraint nc = new NonlinearConstraint(funFn, jacFn, hessFn, lb, ub, kf);
		CanonicalConstraint c = CanonicalConstraint.fromNonlinearConstraint(nc, n);

		assertEquals(1, c.nEq);
		assertEquals(4, c.nIneq);

		EqIneqSplit fSplit = c.fun(x0);
		EqIneqSplit jSplit = c.jac(x0);

		// c_eq = [f[1] - lb[1]]
		RelAbsAssertions.assertRelAbsEquals(f[1] - lb[1], fSplit.eq().getAsDouble(0, 0), TOL);

		// c_ineq order (less | greater | interval-upper | interval-lower):
		//   row 3 less:           f[3] - ub[3]
		//   row 4 greater:        lb[4] - f[4]
		//   row 0 interval-upper: f[0] - ub[0]
		//   row 0 interval-lower: lb[0] - f[0]
		double[] expectedCIneq = {
				f[3] - ub[3], lb[4] - f[4], f[0] - ub[0], lb[0] - f[0]};
		for (int i = 0; i < expectedCIneq.length; ++i) {
			RelAbsAssertions.assertRelAbsEquals(expectedCIneq[i],
					fSplit.ineq().getAsDouble(i, 0), TOL,
					"c_ineq[" + i + "]");
		}

		// J_eq = J[1, :]
		double[][] expectedJEq = {{Jx0.getAsDouble(1, 0), Jx0.getAsDouble(1, 1),
				Jx0.getAsDouble(1, 2)}};
		assertMatrixEquals(expectedJEq, jSplit.eq());

		// J_ineq = vstack(J[3], -J[4], J[0], -J[0])
		double[][] expectedJIneq = {
				{Jx0.getAsDouble(3, 0), Jx0.getAsDouble(3, 1), Jx0.getAsDouble(3, 2)},
				{-Jx0.getAsDouble(4, 0), -Jx0.getAsDouble(4, 1), -Jx0.getAsDouble(4, 2)},
				{Jx0.getAsDouble(0, 0), Jx0.getAsDouble(0, 1), Jx0.getAsDouble(0, 2)},
				{-Jx0.getAsDouble(0, 0), -Jx0.getAsDouble(0, 1), -Jx0.getAsDouble(0, 2)}};
		assertMatrixEquals(expectedJIneq, jSplit.ineq());

		// Hessian-of-Lagrangian: c.hess(x, vEq, vIneq) should equal hess(x, v)
		// where v[i] is reconstructed from (vEq, vIneq) with the canonical
		// sign-flip mask:
		//   v[1]  =  vEq[0]
		//   v[3]  =  vIneq[0]
		//   v[4]  = -vIneq[1]
		//   v[0]  =  vIneq[2] - vIneq[3]
		double[] vEq = {1.5};
		double[] vIneq = {0.7, 0.4, 0.9, 0.1};
		double[] v = new double[m];
		v[1] = vEq[0];
		v[3] = vIneq[0];
		v[4] = -vIneq[1];
		v[0] = vIneq[2] - vIneq[3];
		Matrix expectedH = q.hess(x0, Matrix.Factory.linkToArray(v));
		Matrix actualH = c.hess(x0, vEq, vIneq);
		assertMatrixEquals(expectedH, actualH);

		// keep_feasible per canonical ineq row, in canonical 4-block order
		// (less, greater, interval-upper, interval-lower):
		//   row 3 less:           kf[3] = true
		//   row 4 greater:        kf[4] = false
		//   row 0 interval-upper: kf[0] = true
		//   row 0 interval-lower: kf[0] = true
		assertArrayEquals(new boolean[] {true, false, true, true}, c.keepFeasible);
	}

	@Test
	void testConcatenation() {
		final int n = 4;
		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.1, 0.2, 0.3, 0.4});

		// Source 1: identity-A bounds-style linear constraint with mixed shapes.
		// scipy: Bounds([-1, -inf, -2, 3], [1, inf, inf, 3], [False, False, True, False])
		Matrix A = Matrix.Factory.eye(4, 4);
		double[] lb1 = {-1, Double.NEGATIVE_INFINITY, -2, 3};
		double[] ub1 = {1, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, 3};
		boolean[] kf1 = {false, false, true, false};
		LinearConstraint lc = new LinearConstraint(A, lb1, ub1, kf1);
		// f1 = x0, J1 = I
		double[] f1 = new double[n];
		for (int i = 0; i < n; ++i) f1[i] = x0.getAsDouble(i, 0);

		// Source 2: nonlinear quadratic constraint with mixed shapes.
		// scipy: NonlinearConstraint(fun, [-10, 3, -inf, -inf, -5], [10, 3, inf, 5, inf],
		//                            jac, hess, [True, False, False, True, False])
		final int m = 5;
		QuadraticConstraint q = QuadraticConstraint.deterministic(n, m);
		Matrix fx0 = q.fun(x0);
		Matrix Jx0 = q.jac(x0);
		double[] f2 = new double[m];
		for (int i = 0; i < m; ++i) f2[i] = fx0.getAsDouble(i, 0);
		double[] lb2 = {-10, 3, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, -5};
		double[] ub2 = {10, 3, Double.POSITIVE_INFINITY, 5, Double.POSITIVE_INFINITY};
		boolean[] kf2 = {true, false, false, true, false};
		Function<Matrix, Matrix> funFn = q::fun;
		Function<Matrix, Matrix> jacFn = q::jac;
		BiFunction<Matrix, Matrix, Matrix> hessFn = q::hess;
		NonlinearConstraint nc = new NonlinearConstraint(funFn, jacFn, hessFn, lb2, ub2, kf2);

		// Sparse and dense paths.
		for (boolean sparseJacobian : new boolean[] {false, true}) {
			CanonicalConstraint c1 = CanonicalConstraint.fromLinearConstraint(lc);
			CanonicalConstraint c2 = CanonicalConstraint.fromNonlinearConstraint(nc, n);
			CanonicalConstraint c = CanonicalConstraint.concatenate(
					List.of(c1, c2), sparseJacobian);

			assertEquals(2, c.nEq);
			assertEquals(7, c.nIneq);

			EqIneqSplit fSplit = c.fun(x0);
			EqIneqSplit jSplit = c.jac(x0);

			// c_eq: source-1 eq (row 3 of lc, eq lb=ub=3) gives f1[3] - lb1[3];
			//       source-2 eq (row 1 of nc, eq lb=ub=3) gives f2[1] - lb2[1].
			RelAbsAssertions.assertRelAbsEquals(f1[3] - lb1[3], fSplit.eq().getAsDouble(0, 0), TOL,
					"c_eq[0] (lc row 3)");
			RelAbsAssertions.assertRelAbsEquals(f2[1] - lb2[1], fSplit.eq().getAsDouble(1, 0), TOL,
					"c_eq[1] (nc row 1)");

			// c_ineq from source 1 (in scipy 4-block order): less=[2], greater=[0],
			//   interval-upper=[], interval-lower=[].
			// Wait -- row 0 is lb=-1, ub=1 (interval), but in this test lc rows are:
			//   row 0: lb=-1, ub=1     -> interval (both finite, lb<ub)
			//   row 1: lb=-inf, ub=inf -> dropped
			//   row 2: lb=-2, ub=inf   -> greater (lower-only)
			//   row 3: lb=3, ub=3      -> equality
			// So less=[], greater=[2], interval-upper=[0], interval-lower=[0].
			// c_ineq from source 1 (canonical order):
			//   greater row 2:        lb1[2] - f1[2]
			//   interval-upper row 0: f1[0] - ub1[0]
			//   interval-lower row 0: lb1[0] - f1[0]
			//
			// c_ineq from source 2:
			//   row 3 less:           f2[3] - ub2[3]
			//   row 4 greater:        lb2[4] - f2[4]
			//   row 0 interval-upper: f2[0] - ub2[0]
			//   row 0 interval-lower: lb2[0] - f2[0]
			double[] expectedCIneq = {
					lb1[2] - f1[2], f1[0] - ub1[0], lb1[0] - f1[0],
					f2[3] - ub2[3], lb2[4] - f2[4], f2[0] - ub2[0], lb2[0] - f2[0]};
			for (int i = 0; i < expectedCIneq.length; ++i) {
				RelAbsAssertions.assertRelAbsEquals(expectedCIneq[i],
						fSplit.ineq().getAsDouble(i, 0), TOL,
						"c_ineq[" + i + "]");
			}

			// J_eq: vstack(I[3], J2[1])
			double[][] expectedJEq = new double[2][n];
			expectedJEq[0][3] = 1.0; // I[3]
			for (int j = 0; j < n; ++j) expectedJEq[1][j] = Jx0.getAsDouble(1, j);
			assertMatrixEquals(expectedJEq, jSplit.eq());

			// J_ineq: vstack(-I[2], I[0], -I[0], J2[3], -J2[4], J2[0], -J2[0])
			double[][] expectedJIneq = new double[7][n];
			expectedJIneq[0][2] = -1.0;       // -I[2]
			expectedJIneq[1][0] = 1.0;        //  I[0]
			expectedJIneq[2][0] = -1.0;       // -I[0]
			for (int j = 0; j < n; ++j) {
				expectedJIneq[3][j] = Jx0.getAsDouble(3, j);   //  J2[3]
				expectedJIneq[4][j] = -Jx0.getAsDouble(4, j);  // -J2[4]
				expectedJIneq[5][j] = Jx0.getAsDouble(0, j);   //  J2[0]
				expectedJIneq[6][j] = -Jx0.getAsDouble(0, j);  // -J2[0]
			}
			assertMatrixEquals(expectedJIneq, jSplit.ineq());

			// Hessian: source 1 (linear) contributes 0; source 2 contributes
			// hess(x, v) where v is reconstructed from the source-2 slice of
			// (vEq, vIneq). v_eq has length 2 (one row per source); v_ineq has
			// length 7 (3 from source 1, 4 from source 2).
			double[] vEq = {0.5, 1.5};
			double[] vIneq = {0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4};
			double[] v = new double[m];
			v[1] = vEq[1];
			v[3] = vIneq[3];
			v[4] = -vIneq[4];
			v[0] = vIneq[5] - vIneq[6];
			Matrix expectedH = q.hess(x0, Matrix.Factory.linkToArray(v));
			Matrix actualH = c.hess(x0, vEq, vIneq);
			assertMatrixEquals(expectedH, actualH);

			// keep_feasible (canonical row order across both sources):
			//   source 1 ineqs: greater row 2 (kf[2]=true), interval-upper row 0 (kf[0]=false),
			//                   interval-lower row 0 (kf[0]=false)
			//   source 2 ineqs: less row 3 (kf[3]=true), greater row 4 (kf[4]=false),
			//                   interval-upper row 0 (kf[0]=true), interval-lower row 0 (kf[0]=true)
			assertArrayEquals(new boolean[] {true, false, false, true, false, true, true},
					c.keepFeasible);
		}
	}

	@Test
	void testEmpty() {
		Matrix x = Matrix.Factory.linkToArray(new double[] {1, 2, 3});
		CanonicalConstraint c = CanonicalConstraint.empty(3);
		assertEquals(0, c.nEq);
		assertEquals(0, c.nIneq);

		EqIneqSplit f = c.fun(x);
		assertEquals(0, f.eq().getRowCount());
		assertEquals(0, f.ineq().getRowCount());

		EqIneqSplit J = c.jac(x);
		assertEquals(0, J.eq().getRowCount());
		assertEquals(3, J.eq().getColumnCount());
		assertEquals(0, J.ineq().getRowCount());
		assertEquals(3, J.ineq().getColumnCount());

		Matrix H = c.hess(x, null, null);
		assertEquals(3, H.getRowCount());
		assertEquals(3, H.getColumnCount());
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				assertEquals(0.0, H.getAsDouble(i, j));
			}
		}
	}

	@Test
	void testInitialConstraintsAsCanonical() {
		final int n = 4;
		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 0.4, 0.3, 0.2});

		// Source 1: identity-A linear constraint (acting as bounds with per-row kf).
		Matrix A = Matrix.Factory.eye(4, 4);
		double[] lb1 = {-1, Double.NEGATIVE_INFINITY, -2, 3};
		double[] ub1 = {1, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, 3};
		boolean[] kf1 = {false, false, true, false};
		LinearConstraint lc = new LinearConstraint(A, lb1, ub1, kf1);
		double[] f1 = new double[n];
		for (int i = 0; i < n; ++i) f1[i] = x0.getAsDouble(i, 0);

		// Source 2: nonlinear quadratic constraint.
		final int m = 5;
		QuadraticConstraint q = QuadraticConstraint.deterministic(n, m);
		Matrix fx0 = q.fun(x0);
		Matrix Jx0 = q.jac(x0);
		double[] f2 = new double[m];
		for (int i = 0; i < m; ++i) f2[i] = fx0.getAsDouble(i, 0);
		double[] lb2 = {-10, 3, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, -5};
		double[] ub2 = {10, 3, Double.POSITIVE_INFINITY, 5, Double.POSITIVE_INFINITY};
		boolean[] kf2 = {true, false, false, true, false};
		NonlinearConstraint nc = new NonlinearConstraint(q::fun, q::jac, q::hess, lb2, ub2, kf2);

		for (boolean sparseJacobian : new boolean[] {false, true}) {
			CanonicalConstraint c1 = CanonicalConstraint.fromLinearConstraint(lc);
			CanonicalConstraint c2 = CanonicalConstraint.fromNonlinearConstraint(nc, n);
			CanonicalConstraint.InitialCanonical r =
					CanonicalConstraint.initialConstraintsAsCanonical(
							n, List.of(c1, c2), x0, sparseJacobian);

			// c_eq: same as testConcatenation
			RelAbsAssertions.assertRelAbsEquals(f1[3] - lb1[3], r.cEq().getAsDouble(0, 0), TOL);
			RelAbsAssertions.assertRelAbsEquals(f2[1] - lb2[1], r.cEq().getAsDouble(1, 0), TOL);

			// c_ineq: same as testConcatenation
			double[] expectedCIneq = {
					lb1[2] - f1[2], f1[0] - ub1[0], lb1[0] - f1[0],
					f2[3] - ub2[3], lb2[4] - f2[4], f2[0] - ub2[0], lb2[0] - f2[0]};
			for (int i = 0; i < expectedCIneq.length; ++i) {
				RelAbsAssertions.assertRelAbsEquals(expectedCIneq[i],
						r.cIneq().getAsDouble(i, 0), TOL,
						"c_ineq[" + i + "]");
			}

			// J_eq: same as testConcatenation
			double[][] expectedJEq = new double[2][n];
			expectedJEq[0][3] = 1.0;
			for (int j = 0; j < n; ++j) expectedJEq[1][j] = Jx0.getAsDouble(1, j);
			assertMatrixEquals(expectedJEq, r.JEq());

			// J_ineq: same as testConcatenation
			double[][] expectedJIneq = new double[7][n];
			expectedJIneq[0][2] = -1.0;
			expectedJIneq[1][0] = 1.0;
			expectedJIneq[2][0] = -1.0;
			for (int j = 0; j < n; ++j) {
				expectedJIneq[3][j] = Jx0.getAsDouble(3, j);
				expectedJIneq[4][j] = -Jx0.getAsDouble(4, j);
				expectedJIneq[5][j] = Jx0.getAsDouble(0, j);
				expectedJIneq[6][j] = -Jx0.getAsDouble(0, j);
			}
			assertMatrixEquals(expectedJIneq, r.JIneq());
		}
	}

	@Test
	void testInitialConstraintsAsCanonicalEmpty() {
		final int n = 3;
		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.0, 0.0, 0.0});
		for (boolean sparseJacobian : new boolean[] {false, true}) {
			CanonicalConstraint.InitialCanonical r =
					CanonicalConstraint.initialConstraintsAsCanonical(
							n, List.of(), x0, sparseJacobian);
			assertEquals(0, r.cEq().getRowCount());
			assertEquals(0, r.cIneq().getRowCount());
			assertEquals(0, r.JEq().getRowCount());
			assertEquals(n, r.JEq().getColumnCount());
			assertEquals(0, r.JIneq().getRowCount());
			assertEquals(n, r.JIneq().getColumnCount());
		}
	}

	private static void assertMatrixEquals(double[][] expected, Matrix actual) {
		assertEquals(expected.length, actual.getRowCount(), "row count");
		if (expected.length > 0) {
			assertEquals(expected[0].length, actual.getColumnCount(), "column count");
		}
		for (int i = 0; i < expected.length; ++i) {
			for (int j = 0; j < expected[i].length; ++j) {
				RelAbsAssertions.assertRelAbsEquals(expected[i][j],
						actual.getAsDouble(i, j), TOL, "[" + i + "," + j + "]");
			}
		}
	}

	private static void assertMatrixEquals(Matrix expected, Matrix actual) {
		assertEquals(expected.getRowCount(), actual.getRowCount(), "row count");
		assertEquals(expected.getColumnCount(), actual.getColumnCount(), "column count");
		for (long i = 0; i < expected.getRowCount(); ++i) {
			for (long j = 0; j < expected.getColumnCount(); ++j) {
				RelAbsAssertions.assertRelAbsEquals(
						expected.getAsDouble(i, j), actual.getAsDouble(i, j), TOL,
						"[" + i + "," + j + "]");
			}
		}
	}

	/**
	 * Hand-coded deterministic quadratic constraint
	 * {@code f_i(x) = a_i + A_i.x + 0.5 x^T H_i x} for {@code i = 0..m-1}, with
	 * Jacobian {@code J_i = A_i + H_i x} and Hessian-of-Lagrangian
	 * {@code Sum_i v_i H_i}. Mirrors scipy's
	 * {@code create_quadratic_function} but with fixed coefficients to keep the
	 * test deterministic without a Java-side {@code RandomState(0)} mock.
	 */
	private static final class QuadraticConstraint {
		private final double[] a;
		private final double[][] A;     // m x n
		private final double[][][] H;   // m matrices, each n x n (symmetric)

		private QuadraticConstraint(double[] a, double[][] A, double[][][] H) {
			this.a = a;
			this.A = A;
			this.H = H;
		}

		static QuadraticConstraint deterministic(int n, int m) {
			double[] a = new double[m];
			double[][] A = new double[m][n];
			double[][][] H = new double[m][n][n];
			for (int i = 0; i < m; ++i) {
				a[i] = 1.0 + i;
				for (int j = 0; j < n; ++j) {
					A[i][j] = 0.5 + 0.1 * (i + j);
					for (int k = 0; k <= j; ++k) {
						double v = 0.05 * (i + 1) * ((j == k) ? 2.0 : 1.0);
						H[i][j][k] = v;
						H[i][k][j] = v;
					}
				}
			}
			return new QuadraticConstraint(a, A, H);
		}

		Matrix fun(Matrix x) {
			int m = a.length;
			int n = (int) x.getRowCount();
			Matrix out = Matrix.Factory.zeros(m, 1);
			for (int i = 0; i < m; ++i) {
				double v = a[i];
				for (int j = 0; j < n; ++j) {
					v += A[i][j] * x.getAsDouble(j, 0);
				}
				for (int j = 0; j < n; ++j) {
					for (int k = 0; k < n; ++k) {
						v += 0.5 * H[i][j][k] * x.getAsDouble(j, 0) * x.getAsDouble(k, 0);
					}
				}
				out.setAsDouble(v, i, 0);
			}
			return out;
		}

		Matrix jac(Matrix x) {
			int m = a.length;
			int n = A[0].length;
			Matrix out = Matrix.Factory.zeros(m, n);
			for (int i = 0; i < m; ++i) {
				for (int j = 0; j < n; ++j) {
					double v = A[i][j];
					for (int k = 0; k < n; ++k) {
						v += H[i][j][k] * x.getAsDouble(k, 0);
					}
					out.setAsDouble(v, i, j);
				}
			}
			return out;
		}

		Matrix hess(Matrix x, Matrix v) {
			int m = a.length;
			int n = A[0].length;
			Matrix out = Matrix.Factory.zeros(n, n);
			for (int i = 0; i < m; ++i) {
				double vi = v.getAsDouble(i, 0);
				if (vi == 0.0) continue;
				for (int j = 0; j < n; ++j) {
					for (int k = 0; k < n; ++k) {
						double cur = out.getAsDouble(j, k);
						out.setAsDouble(cur + vi * H[i][j][k], j, k);
					}
				}
			}
			return out;
		}
	}
}
