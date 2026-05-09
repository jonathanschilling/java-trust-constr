package de.labathome.trustconstr;

import java.util.function.BiFunction;
import java.util.function.Function;

import de.labathome.trustconstr.enums.FiniteDifferenceMethod;
import de.labathome.trustconstr.interfaces.Constraint;
import de.labathome.trustconstr.interfaces.Jacobian;
import de.labathome.trustconstr.records.Bounds;
import de.labathome.trustconstr.records.FiniteDifferenceBounds;
import de.labathome.trustconstr.records.FiniteDifferenceOptions;

import de.labathome.trustconstr.matrix.Matrix;

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
 * including the two-sided interval split. Inequality rows are emitted in
 * scipy's 4-block order (less, greater, interval-upper, interval-lower).
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

	/**
	 * Build a nonlinear constraint with explicit constraint Hessian.
	 *
	 * @param fun          constraint function {@code R^n -> R^m}
	 * @param jac          analytic Jacobian {@code R^n -> R^{m x n}}
	 * @param hess         constraint Hessian-of-Lagrangian
	 *                     {@code (x, v) -> Sum v[i] H_{c_i}(x)}; may be {@code null}
	 * @param lb           per-row lower bounds, length {@code m}
	 * @param ub           per-row upper bounds, length {@code m}
	 * @param keepFeasible per-row strict-feasibility flag, length {@code m};
	 *                     {@code null} = all-false
	 */
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
		// Scipy 4-block order: equal | less | greater | interval-upper | interval-lower.
		int eqIdx = 0;
		int ineqIdx = 0;
		for (int i = 0; i < m; ++i) {
			boolean lbFinite = !Double.isInfinite(lb[i]);
			boolean ubFinite = !Double.isInfinite(ub[i]);
			if (lbFinite && ubFinite && lb[i] == ub[i]) {
				eqRows[eqIdx++] = i;
			}
		}
		for (int i = 0; i < m; ++i) {
			boolean lbFinite = !Double.isInfinite(lb[i]);
			boolean ubFinite = !Double.isInfinite(ub[i]);
			if (!lbFinite && ubFinite) {
				ineqRows[ineqIdx] = i;
				ineqSign[ineqIdx] = +1;
				ineqTarget[ineqIdx] = ub[i];
				++ineqIdx;
			}
		}
		for (int i = 0; i < m; ++i) {
			boolean lbFinite = !Double.isInfinite(lb[i]);
			boolean ubFinite = !Double.isInfinite(ub[i]);
			if (lbFinite && !ubFinite) {
				ineqRows[ineqIdx] = i;
				ineqSign[ineqIdx] = -1;
				ineqTarget[ineqIdx] = lb[i];
				++ineqIdx;
			}
		}
		for (int i = 0; i < m; ++i) {
			boolean lbFinite = !Double.isInfinite(lb[i]);
			boolean ubFinite = !Double.isInfinite(ub[i]);
			if (lbFinite && ubFinite && lb[i] != ub[i]) {
				ineqRows[ineqIdx] = i;
				ineqSign[ineqIdx] = +1;
				ineqTarget[ineqIdx] = ub[i];
				++ineqIdx;
			}
		}
		for (int i = 0; i < m; ++i) {
			boolean lbFinite = !Double.isInfinite(lb[i]);
			boolean ubFinite = !Double.isInfinite(ub[i]);
			if (lbFinite && ubFinite && lb[i] != ub[i]) {
				ineqRows[ineqIdx] = i;
				ineqSign[ineqIdx] = -1;
				ineqTarget[ineqIdx] = lb[i];
				++ineqIdx;
			}
		}
	}

	/**
	 * Convenience overload: as {@link #NonlinearConstraint(Function, Function,
	 * BiFunction, double[], double[], boolean[])} with no constraint Hessian.
	 *
	 * @param fun          constraint function
	 * @param jac          analytic Jacobian
	 * @param lb           lower bounds
	 * @param ub           upper bounds
	 * @param keepFeasible per-row strict-feasibility flag
	 */
	public NonlinearConstraint(Function<Matrix, Matrix> fun, Function<Matrix, Matrix> jac,
			double[] lb, double[] ub, boolean[] keepFeasible) {
		this(fun, jac, null, lb, ub, keepFeasible);
	}

	/**
	 * Convenience overload with neither constraint Hessian nor
	 * {@code keepFeasible} flags.
	 *
	 * @param fun constraint function
	 * @param jac analytic Jacobian
	 * @param lb  lower bounds
	 * @param ub  upper bounds
	 */
	public NonlinearConstraint(Function<Matrix, Matrix> fun, Function<Matrix, Matrix> jac,
			double[] lb, double[] ub) {
		this(fun, jac, null, lb, ub, null);
	}

	/**
	 * Construct a {@code NonlinearConstraint} without an analytic Jacobian.
	 * The Jacobian is built lazily by 2-point finite differences on each
	 * call. Mirrors scipy's behaviour when {@code jac} is omitted.
	 *
	 * <p>Only suitable for small-to-medium constraint dimensions: each
	 * {@code jac(x)} call performs {@code n} evaluations of {@code fun}
	 * (where {@code n = x.size()}). For tight problems an analytic
	 * Jacobian remains preferable.
	 */
	/**
	 * @param fun          constraint function
	 * @param lb           lower bounds
	 * @param ub           upper bounds
	 * @param keepFeasible per-row strict-feasibility flag
	 */
	public NonlinearConstraint(Function<Matrix, Matrix> fun,
			double[] lb, double[] ub, boolean[] keepFeasible) {
		this(fun, fdJacobian(fun), null, lb, ub, keepFeasible);
	}

	/**
	 * @param fun constraint function
	 * @param lb  lower bounds
	 * @param ub  upper bounds
	 */
	public NonlinearConstraint(Function<Matrix, Matrix> fun,
			double[] lb, double[] ub) {
		this(fun, fdJacobian(fun), null, lb, ub, null);
	}

	/**
	 * Build a 2-point finite-difference Jacobian closure for a vector-valued
	 * constraint function. Each invocation calls {@code fun(x)} once for the
	 * baseline plus {@code n} more times for the perturbed columns.
	 */
	private static Function<Matrix, Matrix> fdJacobian(Function<Matrix, Matrix> fun) {
		return x -> {
			Matrix f0 = fun.apply(x);
			FiniteDifferenceOptions options = new FiniteDifferenceOptions.FiniteDifferenceOptionsFactory()
					.method(FiniteDifferenceMethod.TWO_POINT)
					.bounds(FiniteDifferenceBounds.unbounded(x.getRowCount()))
					.build();
			return NumDiff.approxDerivative(fun, x, f0, options);
		};
	}

	/** @return defensive copy of the lower bounds */
	public double[] lb() { return lb.clone(); }
	/** @return defensive copy of the upper bounds */
	public double[] ub() { return ub.clone(); }
	/** @return defensive copy of the per-row keep-feasible flags */
	public boolean[] keepFeasible() { return keepFeasible.clone(); }
	/** @return the user-supplied constraint function {@code R^n -> R^m} */
	public Function<Matrix, Matrix> userFun() { return fun; }
	/** @return the user-supplied analytic Jacobian {@code R^n -> R^{m x n}} */
	public Function<Matrix, Matrix> userJac() { return jac; }
	/**
	 * @return the user-supplied constraint Hessian-of-Lagrangian
	 *         {@code (x, v) -> Sum v[i] H_{c_i}(x)}, or {@code null} if none
	 */
	public BiFunction<Matrix, Matrix, Matrix> userHess() { return hess; }
	/** @return the bounds repackaged as a {@link Bounds} record */
	public Bounds bounds() {
		return new Bounds(Matrix.Factory.linkToArray(lb), Matrix.Factory.linkToArray(ub), false);
	}

	/** @return number of canonical equality rows */
	public int nEq() { return eqRows.length; }
	/** @return number of canonical inequality rows */
	public int nIneq() { return ineqRows.length; }

	/**
	 * Per-canonical-inequality-row {@code enforceFeasibility} flags derived
	 * from the user-supplied {@code keep_feasible}. Each canonical ineq row
	 * inherits the kf flag of the original row it came from. Length matches
	 * {@link #nIneq()}.
	 *
	 * @return per-canonical-row strict-feasibility flags
	 */
	public boolean[] enforceFeasibilityIneq() {
		boolean[] out = new boolean[ineqRows.length];
		for (int k = 0; k < ineqRows.length; ++k) {
			out[k] = keepFeasible[ineqRows[k]];
		}
		return out;
	}

	/**
	 * Throws {@link IllegalArgumentException} if any row marked
	 * {@code keepFeasible[i] == true} is violated at the supplied starting
	 * point. Mirrors scipy's strict-feasibility precondition for
	 * {@code keep_feasible=True} rows.
	 *
	 * @param x0 starting iterate ({@code n x 1})
	 */
	public void validateKeepFeasibleAtStart(Matrix x0) {
		boolean any = false;
		for (boolean kf : keepFeasible) {
			if (kf) { any = true; break; }
		}
		if (!any) return;
		Matrix fx = fun.apply(x0);
		for (int i = 0; i < keepFeasible.length; ++i) {
			if (!keepFeasible[i]) continue;
			double v = fx.getAsDouble(i, 0);
			if (v < lb[i] || v > ub[i]) {
				throw new IllegalArgumentException(
						"keep_feasible row " + i + " is violated at x0: "
								+ "lb=" + lb[i] + ", fun(x0)[" + i + "]=" + v + ", ub=" + ub[i]);
			}
		}
	}

	/**
	 * Constraint Hessian-of-Lagrangian {@code Sum_i v[i] * H_{c_i}(x)},
	 * combining equality and inequality multipliers into a single Hessian
	 * contribution.
	 *
	 * @param x     iterate ({@code n x 1})
	 * @param vEq   equality multipliers, length {@link #nEq()}
	 * @param vIneq inequality multipliers, length {@link #nIneq()}
	 * @return summed {@code n x n} constraint Hessian, or {@code null} if no
	 *         analytic Hessian was supplied at construction
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
		Matrix vMat = Matrix.Factory.linkToArray(v);
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
