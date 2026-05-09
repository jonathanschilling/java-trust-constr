package org.scipy.optimize.minimize;

import java.util.function.BiFunction;
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
 * its Jacobian ({@code R^n -> R^{m x n}}). The optional {@code hess} callable
 * is the constraint-Hessian-of-Lagrangian:
 * {@code hess(x, v) -> sum_i v[i] * H_{c_i}(x)}. When supplied, the
 * orchestrator uses it to build the full Lagrangian Hessian
 * {@code H_objective(x) + hess(x, v)}; without it, the orchestrator falls back
 * to the objective Hessian alone, which can hurt convergence on problems where
 * constraint curvature matters at the optimum (e.g. Maratos).
 *
 * <p>Eq/ineq classification follows the same rule as {@link LinearConstraint},
 * including the two-sided interval split.
 */
public class NonlinearConstraint implements Constraint, Jacobian {

	private final Function<Matrix, Matrix> fun;
	private final Function<Matrix, Matrix> jac;
	private final BiFunction<Matrix, Matrix, Matrix> hess;
	private final double[] lb;
	private final double[] ub;
	private final boolean[] keepFeasible;

	private final int[] eqRows;
	private final int[] ineqRows;
	private final int[] ineqSign;
	private final double[] ineqTarget;

	public NonlinearConstraint(Function<Matrix, Matrix> fun, Function<Matrix, Matrix> jac,
			BiFunction<Matrix, Matrix, Matrix> hess,
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
		this.hess = hess;
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
			double[] lb, double[] ub, boolean[] keepFeasible) {
		this(fun, jac, null, lb, ub, keepFeasible);
	}

	public NonlinearConstraint(Function<Matrix, Matrix> fun, Function<Matrix, Matrix> jac,
			double[] lb, double[] ub) {
		this(fun, jac, null, lb, ub, null);
	}

	public double[] lb() { return lb.clone(); }
	public double[] ub() { return ub.clone(); }
	public boolean[] keepFeasible() { return keepFeasible.clone(); }
	public Bounds bounds() {
		return new Bounds(UjmpBridge.arrayToCol(lb), UjmpBridge.arrayToCol(ub), false);
	}

	public int nEq() { return eqRows.length; }
	public int nIneq() { return ineqRows.length; }

	/**
	 * Constraint Hessian-of-Lagrangian {@code sum_i v[i] * H_{c_i}(x)},
	 * combining equality and inequality multipliers into a single Hessian
	 * contribution. Returns {@code null} if no Hessian was supplied.
	 *
	 * <p>{@code vEq} has length {@link #nEq()} and {@code vIneq} length
	 * {@link #nIneq()}. The values are mapped back to the original constraint
	 * rows (with appropriate sign for one-sided inequalities) before invoking
	 * the user-provided {@code hess(x, v)}.
	 */
	public Matrix lagrangianContribution(Matrix x, double[] vEq, double[] vIneq) {
		if (hess == null) {
			return null;
		}
		int m = lb.length;
		double[] v = new double[m];
		for (int e = 0; e < eqRows.length; ++e) {
			v[eqRows[e]] += vEq[e];
		}
		for (int k = 0; k < ineqRows.length; ++k) {
			v[ineqRows[k]] += ineqSign[k] * vIneq[k];
		}
		Matrix vMat = UjmpBridge.arrayToCol(v);
		return hess.apply(x, vMat);
	}

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
