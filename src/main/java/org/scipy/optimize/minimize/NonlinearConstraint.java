package org.scipy.optimize.minimize;

import java.util.function.Function;

import org.scipy.optimize.minimize.interfaces.Constraint;
import org.scipy.optimize.minimize.interfaces.Jacobian;
import org.scipy.optimize.minimize.records.Bounds;
import org.scipy.optimize.minimize.sparse.UjmpBridge;
import org.ujmp.core.Matrix;

/**
 * Nonlinear constraint of the form
 *
 * <pre>
 *     lb &lt;= fun(x) &lt;= ub
 * </pre>
 *
 * Counterpart to {@code scipy.optimize.NonlinearConstraint}
 * ({@code scipy/optimize/_constraints.py:22}).
 *
 * <p>{@code fun} is the constraint function {@code R^n -> R^m}, {@code jac}
 * its Jacobian ({@code R^n -> R^{m x n}}). The Hessian-of-Lagrangian callable
 * is left out of this iteration — analytic Hessians for nonlinear constraints
 * are added by passing a {@code BFGS}/{@code SR1} update strategy through the
 * top-level {@code MinimizeTrustConstr.minimizeTrustConstr(...)} call.
 *
 * <p>Eq/ineq classification follows the same rule as {@link LinearConstraint},
 * including the two-sided interval split.
 */
public class NonlinearConstraint implements Constraint, Jacobian {

	private final Function<Matrix, Matrix> fun;
	private final Function<Matrix, Matrix> jac;
	private final double[] lb;
	private final double[] ub;
	private final boolean[] keepFeasible;

	private final int[] eqRows;
	private final int[] ineqRows;
	private final int[] ineqSign;
	private final double[] ineqTarget;

	public NonlinearConstraint(Function<Matrix, Matrix> fun, Function<Matrix, Matrix> jac,
			double[] lb, double[] ub, boolean[] keepFeasible) {
		if (lb.length != ub.length) {
			throw new IllegalArgumentException("lb and ub must have the same length");
		}
		int m = lb.length;
		if (keepFeasible == null) {
			keepFeasible = new boolean[m];
		} else if (keepFeasible.length != m) {
			throw new IllegalArgumentException("keepFeasible must have length lb.length");
		}
		this.fun = fun;
		this.jac = jac;
		this.lb = lb.clone();
		this.ub = ub.clone();
		this.keepFeasible = keepFeasible.clone();

		int nEq = 0;
		int nIneq = 0;
		for (int i = 0; i < m; ++i) {
			boolean lbFinite = !Double.isInfinite(lb[i]);
			boolean ubFinite = !Double.isInfinite(ub[i]);
			if (lbFinite && ubFinite && lb[i] == ub[i]) {
				++nEq;
			} else if (lbFinite && ubFinite) {
				nIneq += 2;
			} else if (lbFinite || ubFinite) {
				++nIneq;
			}
		}
		this.eqRows = new int[nEq];
		this.ineqRows = new int[nIneq];
		this.ineqSign = new int[nIneq];
		this.ineqTarget = new double[nIneq];
		int eqIdx = 0;
		int ineqIdx = 0;
		for (int i = 0; i < m; ++i) {
			boolean lbFinite = !Double.isInfinite(lb[i]);
			boolean ubFinite = !Double.isInfinite(ub[i]);
			if (lbFinite && ubFinite && lb[i] == ub[i]) {
				eqRows[eqIdx++] = i;
			} else if (lbFinite && ubFinite) {
				ineqRows[ineqIdx] = i;
				ineqSign[ineqIdx] = +1;
				ineqTarget[ineqIdx] = ub[i];
				++ineqIdx;
				ineqRows[ineqIdx] = i;
				ineqSign[ineqIdx] = -1;
				ineqTarget[ineqIdx] = lb[i];
				++ineqIdx;
			} else if (lbFinite) {
				ineqRows[ineqIdx] = i;
				ineqSign[ineqIdx] = -1;
				ineqTarget[ineqIdx] = lb[i];
				++ineqIdx;
			} else if (ubFinite) {
				ineqRows[ineqIdx] = i;
				ineqSign[ineqIdx] = +1;
				ineqTarget[ineqIdx] = ub[i];
				++ineqIdx;
			}
		}
	}

	public NonlinearConstraint(Function<Matrix, Matrix> fun, Function<Matrix, Matrix> jac,
			double[] lb, double[] ub) {
		this(fun, jac, lb, ub, null);
	}

	public double[] lb() { return lb.clone(); }
	public double[] ub() { return ub.clone(); }
	public boolean[] keepFeasible() { return keepFeasible.clone(); }
	public Bounds bounds() {
		return new Bounds(UjmpBridge.arrayToCol(lb), UjmpBridge.arrayToCol(ub), false);
	}

	public int nEq() { return eqRows.length; }
	public int nIneq() { return ineqRows.length; }

	@Override
	public Matrix constrEq(Matrix x) {
		Matrix fx = fun.apply(x);
		Matrix out = Matrix.Factory.zeros(eqRows.length, 1);
		for (int e = 0; e < eqRows.length; ++e) {
			int i = eqRows[e];
			out.setAsDouble(fx.getAsDouble(i, 0) - lb[i], e, 0);
		}
		return out;
	}

	@Override
	public Matrix constrIneq(Matrix x) {
		Matrix fx = fun.apply(x);
		Matrix out = Matrix.Factory.zeros(ineqRows.length, 1);
		for (int k = 0; k < ineqRows.length; ++k) {
			int i = ineqRows[k];
			double signed = ineqSign[k] * (fx.getAsDouble(i, 0) - ineqTarget[k]);
			out.setAsDouble(signed, k, 0);
		}
		return out;
	}

	@Override
	public Matrix jacEq(Matrix x) {
		Matrix jx = jac.apply(x);
		return rowSelection(jx, eqRows, +1);
	}

	@Override
	public Matrix jacIneq(Matrix x) {
		Matrix jx = jac.apply(x);
		Matrix out = Matrix.Factory.zeros(ineqRows.length, jx.getColumnCount());
		for (int k = 0; k < ineqRows.length; ++k) {
			int i = ineqRows[k];
			double sign = ineqSign[k];
			for (int j = 0; j < jx.getColumnCount(); ++j) {
				out.setAsDouble(sign * jx.getAsDouble(i, j), k, j);
			}
		}
		return out;
	}

	private Matrix rowSelection(Matrix m, int[] rows, int sign) {
		Matrix out = Matrix.Factory.zeros(rows.length, m.getColumnCount());
		for (int k = 0; k < rows.length; ++k) {
			int i = rows[k];
			for (int j = 0; j < m.getColumnCount(); ++j) {
				out.setAsDouble(sign * m.getAsDouble(i, j), k, j);
			}
		}
		return out;
	}
}
