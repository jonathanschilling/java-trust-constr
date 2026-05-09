package de.labathome.optimization;

import java.util.function.BiFunction;
import java.util.function.Function;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.scipy.optimize.minimize.CombinedConstraint;
import org.scipy.optimize.minimize.LinearConstraint;
import org.scipy.optimize.minimize.NonlinearConstraint;
import org.scipy.optimize.minimize.matrix.Matrix;


/**
 * Java translation of scipy
 * {@code _trustregion_constr/tests/test_canonical_constraint.py::test_concatenation}.
 *
 * <p>Stacks a {@link LinearConstraint} with mixed equality/inequality rows and
 * a {@link NonlinearConstraint} with the same shape, and verifies that
 * {@link CombinedConstraint} produces the correct vertically-stacked
 * {@code (constrEq, constrIneq, jacEq, jacIneq)} and that
 * {@link CombinedConstraint#lagrangianContribution} slices the multipliers
 * per source before delegating to each source's Hessian.
 *
 * <p>Each source emits inequality rows in scipy's 4-block order (less,
 * greater, interval-upper, interval-lower). {@link CombinedConstraint} then
 * stacks all eqs source-by-source, then all ineqs source-by-source.
 */
class TestCombinedConstraint {

	private static final double TOL = 1.0e-12;
	private static final double NEG_INF = Double.NEGATIVE_INFINITY;
	private static final double POS_INF = Double.POSITIVE_INFINITY;

	@Test
	void concatenateLinearAndNonlinear() {
		// Source 1: linear, identity Jacobian; bounds chosen to give 1 eq + 3 ineq:
		//   row 0: [-1, 1] interval  -> 2 ineq (interval-upper, interval-lower)
		//   row 1: free (-inf, inf)
		//   row 2: [-2, inf] lower   -> 1 ineq (greater)
		//   row 3: [3, 3] equality   -> 1 eq
		// nEq = 1, nIneq = 3 emitted in scipy 4-block order: [row2 greater, row0 ub, row0 lb].
		Matrix A1 = Matrix.Factory.eye(4, 4);
		LinearConstraint c1 = new LinearConstraint(A1,
				new double[] {-1, NEG_INF, -2, 3},
				new double[] { 1, POS_INF, POS_INF, 3});
		Assertions.assertEquals(1, c1.nEq());
		Assertions.assertEquals(3, c1.nIneq());

		// Source 2: nonlinear; quadratic with a constant Hessian we can verify
		// against. Set m = 5 with the scipy-test bound shape:
		//   lb2 = [-10, 3, -inf, -inf, -5]; ub2 = [10, 3, inf, 5, inf]
		// Row 0: interval -> 2 ineq, row 1: equality -> 1 eq, row 2: free,
		// row 3: upper -> 1 ineq, row 4: lower -> 1 ineq.
		// nEq = 1, nIneq = 4.
		final double[][] Hcoef = {
				{1.0, 0.5, 0.0, 0.0},
				{0.5, 2.0, 0.5, 0.0},
				{0.0, 0.5, 3.0, 0.5},
				{0.0, 0.0, 0.5, 4.0},
		};
		Function<Matrix, Matrix> f2 = x -> {
			double a = x.getAsDouble(0, 0);
			double b = x.getAsDouble(1, 0);
			double c = x.getAsDouble(2, 0);
			double d = x.getAsDouble(3, 0);
			// Each component is x_i^2 / 2 (well, just produce scalar values dependent on x).
			return Matrix.Factory.linkToArray(new double[] {
					a * a + b, b + c, c, d * d + b, a + d});
		};
		Function<Matrix, Matrix> j2 = x -> {
			double a = x.getAsDouble(0, 0);
			double d = x.getAsDouble(3, 0);
			return Matrix.Factory.linkToArray(new double[][] {
					{2 * a, 1, 0, 0},
					{0, 1, 1, 0},
					{0, 0, 1, 0},
					{0, 1, 0, 2 * d},
					{1, 0, 0, 1}});
		};
		// Constraint Hessian-of-Lagrangian: sum_i v[i] * H_{c_i}.
		// For the rows above:
		//   c_0 has H = diag(2, 0, 0, 0)
		//   c_1, c_2 have H = 0
		//   c_3 has H = diag(0, 0, 0, 2)
		//   c_4 has H = 0
		BiFunction<Matrix, Matrix, Matrix> h2 = (x, v) -> {
			double v0 = v.getAsDouble(0, 0);
			double v3 = v.getAsDouble(3, 0);
			return Matrix.Factory.linkToArray(new double[][] {
					{2 * v0, 0, 0, 0},
					{0, 0, 0, 0},
					{0, 0, 0, 0},
					{0, 0, 0, 2 * v3}});
		};
		NonlinearConstraint c2 = new NonlinearConstraint(f2, j2, h2,
				new double[] {-10, 3, NEG_INF, NEG_INF, -5},
				new double[] { 10, 3, POS_INF,    5,    POS_INF},
				null);
		Assertions.assertEquals(1, c2.nEq());
		Assertions.assertEquals(4, c2.nIneq());

		// Combined: nEq = 2, nIneq = 7.
		CombinedConstraint cc = new CombinedConstraint(new Object[] {c1, c2}, 4);
		Assertions.assertEquals(2, cc.nEq());
		Assertions.assertEquals(7, cc.nIneq());

		// Suppress check
		Assertions.assertNotNull(Hcoef);

		Matrix x0 = Matrix.Factory.linkToArray(new double[] {0.5, 0.4, 0.3, 0.2});

		// constrEq concatenates [c1.eq, c2.eq] with c1.eq from row 3 = (x[3] - 3)
		// = (0.2 - 3) = -2.8, and c2.eq from row 1 = f2[1] - 3 = (0.4 + 0.3) - 3 = -2.3.
		RelAbsAssertions.assertArrayRelAbsEquals(
				new double[] {-2.8, -2.3},
				LinAlgArr(cc.constrEq(x0)), TOL);

		// constrIneq concatenates [c1.ineq, c2.ineq]:
		//   c1 in scipy 4-block order [greater, interval-upper, interval-lower]:
		//        row 2 lb (greater):  -1*(x[2] - (-2)) = -2.3
		//        row 0 ub (interval): x[0] - 1 = -0.5
		//        row 0 lb (interval): -1*(x[0] - (-1)) = -1.5
		//   c2 in scipy 4-block order [less, greater, interval-upper, interval-lower]:
		//        row 3 ub (less):     f2[3] - 5 = (0.04 + 0.4) - 5 = -4.56
		//        row 4 lb (greater):  -1*(f2[4] - (-5)) = -1*((0.5 + 0.2) + 5) = -5.7
		//        row 0 ub (interval): f2[0] - 10 = (0.25 + 0.4) - 10 = -9.35
		//        row 0 lb (interval): -1*(f2[0] - (-10)) = -10.65
		RelAbsAssertions.assertArrayRelAbsEquals(
				new double[] {-2.3, -0.5, -1.5, -4.56, -5.7, -9.35, -10.65},
				LinAlgArr(cc.constrIneq(x0)), TOL);

		// Hessian-of-Lagrangian: sum over c1.contribution (zero, linear) and
		// c2.contribution. Pick vEq = [0.7, 0.9] and vIneq sized 7 in the
		// combined ineq order [c1: row2 greater, row0 ub, row0 lb;
		//                      c2: row3 less, row4 greater, row0 ub, row0 lb].
		// Set c2's row0 ub mult = 0.5 (combined index 5) and c2's row4 greater
		// mult = 0.4 (combined index 4); the rest are 0.
		// c1 contributes zero (linear). c2's slice: vEqSlice = [0.9],
		// vIneqSlice = [0, 0.4, 0.5, 0]. Pack into m=5 multiplier vector per
		// scipy mapping: v[1] += 0.9 (eq), v[3] += +1*0 (row 3 less),
		// v[4] += -1*0.4 = -0.4 (row 4 greater), v[0] += +1*0.5 (row 0 ub) +
		// (-1)*0 (row 0 lb) = 0.5.
		// Hessian = 0.5*H_0 + 0.9*H_1 + 0*H_3 + (-0.4)*H_4 = 0.5*diag(2,0,0,0)
		// = diag(1, 0, 0, 0).
		double[] vEq = {0.7, 0.9};
		double[] vIneq = {0, 0, 0, 0, 0.4, 0.5, 0};
		Matrix HLag = cc.lagrangianContribution(x0, vEq, vIneq);
		Assertions.assertNotNull(HLag);
		RelAbsAssertions.assertArrayRelAbsEquals(
				new double[][] {
						{1.0, 0, 0, 0},
						{0,   0, 0, 0},
						{0,   0, 0, 0},
						{0,   0, 0, 0}},
				HLag.toDoubleArray(), TOL);
	}

	@Test
	void jacEqStaysSparseWhenAllSourcesAreSparse() {
		// Two linear constraints, both with sparse A. The combined jacEq
		// should auto-detect sparsity and allocate a sparse output rather
		// than a dense one.
		org.scipy.optimize.minimize.matrix.SparseMatrix A1 = org.scipy.optimize.minimize.matrix.SparseMatrix.Factory.zeros(1, 4);
		A1.setAsDouble(1.0, 0, 0);
		A1.setAsDouble(2.0, 0, 3);
		LinearConstraint c1 = new LinearConstraint(A1,
				new double[] {0.0}, new double[] {0.0});

		org.scipy.optimize.minimize.matrix.SparseMatrix A2 = org.scipy.optimize.minimize.matrix.SparseMatrix.Factory.zeros(1, 4);
		A2.setAsDouble(3.0, 0, 1);
		LinearConstraint c2 = new LinearConstraint(A2,
				new double[] {0.0}, new double[] {0.0});

		CombinedConstraint cc = new CombinedConstraint(new Object[] {c1, c2}, 4);
		Matrix x = Matrix.Factory.linkToArray(new double[] {1, 1, 1, 1});
		Matrix Jeq = cc.jacEq(x);
		Assertions.assertTrue(Jeq.isSparse(),
				"jacEq should be sparse when all sources are sparse");
		Assertions.assertEquals(2, Jeq.getRowCount());
		Assertions.assertEquals(1.0, Jeq.getAsDouble(0, 0), TOL);
		Assertions.assertEquals(2.0, Jeq.getAsDouble(0, 3), TOL);
		Assertions.assertEquals(3.0, Jeq.getAsDouble(1, 1), TOL);
	}

	@Test
	void jacEqIsDenseWhenAnySourceIsDense() {
		// Mixed sources: one sparse, one dense. Combined output should be
		// dense (the dense source's part can't be embedded into a sparse
		// allocator without extra copying -- auto-detect picks the lower
		// common denominator).
		org.scipy.optimize.minimize.matrix.SparseMatrix A1 = org.scipy.optimize.minimize.matrix.SparseMatrix.Factory.zeros(1, 4);
		A1.setAsDouble(1.0, 0, 0);
		LinearConstraint c1 = new LinearConstraint(A1,
				new double[] {0.0}, new double[] {0.0});

		Matrix A2dense = Matrix.Factory.linkToArray(new double[][] {{2, 3, 4, 5}});
		LinearConstraint c2 = new LinearConstraint(A2dense,
				new double[] {0.0}, new double[] {0.0});

		CombinedConstraint cc = new CombinedConstraint(new Object[] {c1, c2}, 4);
		Matrix x = Matrix.Factory.linkToArray(new double[] {1, 1, 1, 1});
		Matrix Jeq = cc.jacEq(x);
		Assertions.assertFalse(Jeq.isSparse(),
				"jacEq should be dense when any source is dense");
		Assertions.assertEquals(2, Jeq.getRowCount());
		Assertions.assertEquals(1.0, Jeq.getAsDouble(0, 0), TOL);
		Assertions.assertEquals(2.0, Jeq.getAsDouble(1, 0), TOL);
		Assertions.assertEquals(5.0, Jeq.getAsDouble(1, 3), TOL);
	}

	@Test
	void emptySourcesYieldZeroDimensionConstraint() {
		CombinedConstraint cc = new CombinedConstraint(new Object[0], 3);
		Assertions.assertEquals(0, cc.nEq());
		Assertions.assertEquals(0, cc.nIneq());

		Matrix x = Matrix.Factory.linkToArray(new double[] {1, 2, 3});
		Assertions.assertEquals(0, cc.constrEq(x).getRowCount());
		Assertions.assertEquals(0, cc.constrIneq(x).getRowCount());
		Assertions.assertEquals(0, cc.jacEq(x).getRowCount());
		Assertions.assertEquals(3, cc.jacEq(x).getColumnCount());
		Assertions.assertEquals(0, cc.jacIneq(x).getRowCount());
		Assertions.assertEquals(3, cc.jacIneq(x).getColumnCount());
	}

	private static double[] LinAlgArr(Matrix col) {
		int n = (int) col.getRowCount();
		double[] a = new double[n];
		for (int i = 0; i < n; ++i) {
			a[i] = col.getAsDouble(i, 0);
		}
		return a;
	}
}
