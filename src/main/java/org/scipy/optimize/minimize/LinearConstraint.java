package org.scipy.optimize.minimize;

import org.scipy.optimize.minimize.interfaces.Constraint;
import org.scipy.optimize.minimize.interfaces.Jacobian;
import org.scipy.optimize.minimize.records.Bounds;
import org.scipy.optimize.minimize.sparse.CSRMatrix;
import org.scipy.optimize.minimize.sparse.UjmpBridge;
import org.ujmp.core.Matrix;

/**
 * Linear constraint of the form
 *
 * <pre>
 *     lb &lt;= A x &lt;= ub
 * </pre>
 *
 * Counterpart to {@code scipy.optimize.LinearConstraint}
 * ({@code scipy/optimize/_constraints.py:139}). Stores the constraint matrix
 * {@code A}, the bounds, and the per-row {@code keepFeasible} flag.
 *
 * <p>The Java port distinguishes equality and inequality at the constraint
 * level — this class implements both {@link Constraint} (constraint values)
 * and {@link Jacobian} (analytic Jacobian, which for a linear constraint is
 * just {@code A}).
 *
 * <p>Each row {@code i} of {@code A} is classified as:
 * <ul>
 *   <li>Equality if {@code lb[i] == ub[i]} (and finite). Contributes one row
 *       to {@code constrEq} / {@code jacEq} with value {@code A[i,:] x - lb[i]}.</li>
 *   <li>Lower-only if {@code ub[i] == +∞}. Contributes one row to
 *       {@code constrIneq} / {@code jacIneq}: {@code lb[i] - A[i,:] x}, sign
 *       flipped so the convention {@code constrIneq(x) <= 0} holds.</li>
 *   <li>Upper-only if {@code lb[i] == -∞}. Contributes one row to
 *       {@code constrIneq} / {@code jacIneq}: {@code A[i,:] x - ub[i]}.</li>
 *   <li>Two-sided interval ({@code lb[i] < ub[i]}, both finite). Contributes
 *       <em>two</em> rows to {@code constrIneq} / {@code jacIneq}: an
 *       upper-side row {@code A[i,:] x - ub[i]} and a lower-side row
 *       {@code lb[i] - A[i,:] x}. Mirrors scipy's
 *       {@code _canonical_constraints} row split.</li>
 * </ul>
 */
public class LinearConstraint implements Constraint, Jacobian {

	private final Matrix A;
	private final double[] lb;
	private final double[] ub;
	private final boolean[] keepFeasible;

	private final int[] eqRows;
	private final int[] ineqRows;
	/** sign[i] = +1 if upper-only ({@code A[i,:] x - ub[i]}), -1 if lower-only ({@code lb[i] - A[i,:] x}) */
	private final int[] ineqSign;
	/** target value for each ineq row: ub for upper-only, -lb for lower-only (so {@code sign*A x + target <= 0}) */
	private final double[] ineqTarget;

	public LinearConstraint(Matrix a, double[] lb, double[] ub, boolean[] keepFeasible) {
		long m = a.getRowCount();
		if (lb.length != m || ub.length != m) {
			throw new IllegalArgumentException("lb and ub must have length A.rows()");
		}
		if (keepFeasible == null) {
			keepFeasible = new boolean[(int) m];
		} else if (keepFeasible.length != m) {
			throw new IllegalArgumentException("keepFeasible must have length A.rows()");
		}
		this.A = a;
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
				// Two-sided interval: emits two ineq rows (upper, lower).
				nIneq += 2;
			} else if (lbFinite || ubFinite) {
				++nIneq;
			}
			// both infinite: no constraint, drop
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
				// Two-sided: emit upper-side row, then lower-side row.
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

	public LinearConstraint(Matrix a, double[] lb, double[] ub) {
		this(a, lb, ub, null);
	}

	public Matrix getA() { return A; }
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
		Matrix ax = A.mtimes(x);
		Matrix out = Matrix.Factory.zeros(eqRows.length, 1);
		for (int e = 0; e < eqRows.length; ++e) {
			int i = eqRows[e];
			out.setAsDouble(ax.getAsDouble(i, 0) - lb[i], e, 0);
		}
		return out;
	}

	@Override
	public Matrix constrIneq(Matrix x) {
		Matrix ax = A.mtimes(x);
		Matrix out = Matrix.Factory.zeros(ineqRows.length, 1);
		for (int k = 0; k < ineqRows.length; ++k) {
			int i = ineqRows[k];
			// sign * A[i,:] x - sign * target  (so for upper-only: A x - ub; for lower-only: -(A x) + lb = -(A x - lb))
			double signed = ineqSign[k] * (ax.getAsDouble(i, 0) - ineqTarget[k]);
			out.setAsDouble(signed, k, 0);
		}
		return out;
	}

	@Override
	public Matrix jacEq(Matrix x) {
		return rowSelection(eqRows, +1);
	}

	@Override
	public Matrix jacIneq(Matrix x) {
		Matrix out = Matrix.Factory.zeros(ineqRows.length, A.getColumnCount());
		for (int k = 0; k < ineqRows.length; ++k) {
			int i = ineqRows[k];
			double sign = ineqSign[k];
			for (int j = 0; j < A.getColumnCount(); ++j) {
				out.setAsDouble(sign * A.getAsDouble(i, j), k, j);
			}
		}
		return out;
	}

	private Matrix rowSelection(int[] rows, int sign) {
		Matrix out = Matrix.Factory.zeros(rows.length, A.getColumnCount());
		for (int k = 0; k < rows.length; ++k) {
			int i = rows[k];
			for (int j = 0; j < A.getColumnCount(); ++j) {
				out.setAsDouble(sign * A.getAsDouble(i, j), k, j);
			}
		}
		return out;
	}

	/** Sparse Jacobian for equality rows: a {@link CSRMatrix} view of the selected rows of {@code A}. */
	public CSRMatrix jacEqCSR() {
		return UjmpBridge.toCSR(jacEq(null));
	}

	/** Sparse Jacobian for inequality rows: a {@link CSRMatrix} view (with appropriate sign flips). */
	public CSRMatrix jacIneqCSR() {
		return UjmpBridge.toCSR(jacIneq(null));
	}
}
