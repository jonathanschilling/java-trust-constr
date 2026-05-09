package org.scipy.optimize.minimize;

import org.scipy.optimize.minimize.interfaces.Constraint;
import org.scipy.optimize.minimize.interfaces.Jacobian;
import org.scipy.optimize.minimize.records.Bounds;
import org.scipy.optimize.minimize.sparse.CSRMatrix;
import org.scipy.optimize.minimize.sparse.UjmpBridge;
import org.ujmp.core.Matrix;
import org.ujmp.core.SparseMatrix;

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
 * <p>Each row {@code i} of {@code A} is classified as one of four categories:
 * <ul>
 *   <li>{@code equal}: {@code lb[i] == ub[i]} (finite). Contributes one row
 *       to {@code constrEq} / {@code jacEq}: {@code A[i,:] x - lb[i]}.</li>
 *   <li>{@code less}: {@code lb[i] == -∞}, {@code ub[i] < ∞} (upper-only).
 *       Contributes one ineq row {@code A[i,:] x - ub[i]} (sign +1).</li>
 *   <li>{@code greater}: {@code ub[i] == +∞}, {@code lb[i] > -∞} (lower-only).
 *       Contributes one ineq row {@code lb[i] - A[i,:] x} (sign -1, target lb).</li>
 *   <li>{@code interval}: both finite, {@code lb[i] < ub[i]}. Contributes
 *       <em>two</em> ineq rows: an upper-side row and a lower-side row.</li>
 * </ul>
 *
 * <p>The {@code constrIneq} / {@code jacIneq} output is grouped scipy-style
 * (mirrors {@code canonical_constraint.py:_interval_to_canonical}): all
 * {@code less} rows first, then all {@code greater} rows, then all
 * {@code interval}-upper rows, then all {@code interval}-lower rows. Within
 * each block the original row order is preserved.
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
		// Scipy 4-block order: equal | less | greater | interval-upper | interval-lower.
		int eqIdx = 0;
		int ineqIdx = 0;
		// Pass 1: equality rows (lb == ub).
		for (int i = 0; i < m; ++i) {
			boolean lbFinite = !Double.isInfinite(lb[i]);
			boolean ubFinite = !Double.isInfinite(ub[i]);
			if (lbFinite && ubFinite && lb[i] == ub[i]) {
				eqRows[eqIdx++] = i;
			}
		}
		// Pass 2: less (lb=-inf, ub<inf) — sign +1, target ub.
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
		// Pass 3: greater (ub=+inf, lb>-inf) — sign -1, target lb.
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
		// Pass 4: interval-upper (both finite, lb<ub) — sign +1, target ub.
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
		// Pass 5: interval-lower (both finite, lb<ub) — sign -1, target lb.
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

	public LinearConstraint(Matrix a, double[] lb, double[] ub) {
		this(a, lb, ub, null);
	}

	/**
	 * Build a {@code LinearConstraint} from variable {@link Bounds}, i.e. the
	 * trivial linear constraint {@code lb <= x <= ub} with {@code A = I}.
	 *
	 * Mirrors scipy's promotion of {@code Bounds} to a canonical constraint
	 * via {@code PreparedConstraint(bounds, ...)}.
	 *
	 * @param bounds variable bounds — {@code lb} and {@code ub} are
	 *               {@code n x 1} column matrices
	 * @return identity-Jacobian constraint that enforces {@code bounds}
	 */
	public static LinearConstraint fromBounds(Bounds bounds) {
		long n = bounds.lb().getRowCount();
		Matrix identity = Matrix.Factory.eye(n, n);
		double[] lbArr = new double[(int) n];
		double[] ubArr = new double[(int) n];
		for (int i = 0; i < n; ++i) {
			lbArr[i] = bounds.lb().getAsDouble(i, 0);
			ubArr[i] = bounds.ub().getAsDouble(i, 0);
		}
		// Propagate keep_feasible from Bounds to per-row keepFeasible — the
		// scalar flag on Bounds applies uniformly to every variable.
		boolean[] keepFeasible = new boolean[(int) n];
		if (bounds.keepFeasible()) {
			java.util.Arrays.fill(keepFeasible, true);
		}
		return new LinearConstraint(identity, lbArr, ubArr, keepFeasible);
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

	/**
	 * Per-canonical-inequality-row {@code enforceFeasibility} flags derived
	 * from the user-supplied {@code keep_feasible}. Each canonical ineq row
	 * inherits the kf flag of the original row it came from
	 * ({@link #ineqRows}). Length matches {@link #nIneq()} and is suitable
	 * to pass directly to {@code TrustRegionInteriorPoint}'s
	 * {@code enforceFeasibility} parameter.
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
	 * {@code keep_feasible=True} rows: the algorithm will not enforce
	 * intermediate-iterate feasibility for these rows if the start is
	 * already infeasible there.
	 */
	public void validateKeepFeasibleAtStart(Matrix x0) {
		boolean any = false;
		for (boolean kf : keepFeasible) {
			if (kf) { any = true; break; }
		}
		if (!any) return;
		Matrix ax = A.mtimes(x0);
		for (int i = 0; i < keepFeasible.length; ++i) {
			if (!keepFeasible[i]) continue;
			double v = ax.getAsDouble(i, 0);
			if (v < lb[i] || v > ub[i]) {
				throw new IllegalArgumentException(
						"keep_feasible row " + i + " is violated at x0: "
								+ "lb=" + lb[i] + ", value=" + v + ", ub=" + ub[i]);
			}
		}
	}

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
		// Sign array of all +1: equality rows pass through unmodified.
		int[] signs = new int[eqRows.length];
		for (int i = 0; i < signs.length; ++i) signs[i] = +1;
		return rowSelection(eqRows, signs);
	}

	@Override
	public Matrix jacIneq(Matrix x) {
		return rowSelection(ineqRows, ineqSign);
	}

	/**
	 * Build the row-selected Jacobian (with optional per-row sign flips for
	 * one-sided lower-bound rows). Preserves sparsity: when {@link #A} is a
	 * UJMP sparse matrix, the output is a sparse {@link SparseMatrix}, which
	 * lets {@code Projections.projections} route through the AUGMENTED_SYSTEM
	 * path automatically.
	 */
	private Matrix rowSelection(int[] rows, int[] signs) {
		long cols = A.getColumnCount();
		if (A.isSparse()) {
			SparseMatrix out = SparseMatrix.Factory.zeros(rows.length, cols);
			for (int k = 0; k < rows.length; ++k) {
				int i = rows[k];
				int sign = signs[k];
				for (int j = 0; j < cols; ++j) {
					double v = A.getAsDouble(i, j);
					if (v != 0.0) {
						out.setAsDouble(sign * v, k, j);
					}
				}
			}
			return out;
		}
		Matrix out = Matrix.Factory.zeros(rows.length, cols);
		for (int k = 0; k < rows.length; ++k) {
			int i = rows[k];
			int sign = signs[k];
			for (int j = 0; j < cols; ++j) {
				out.setAsDouble(sign * A.getAsDouble(i, j), k, j);
			}
		}
		return out;
	}

	/**
	 * Sparse Jacobian for equality rows: a {@link CSRMatrix} built directly
	 * from the row-selected non-zeros of {@code A}, without going through a
	 * dense intermediate.
	 */
	public CSRMatrix jacEqCSR() {
		int[] signs = new int[eqRows.length];
		for (int i = 0; i < signs.length; ++i) signs[i] = +1;
		return rowSelectionCSR(eqRows, signs);
	}

	/** Sparse Jacobian for inequality rows: a {@link CSRMatrix} (with appropriate sign flips). */
	public CSRMatrix jacIneqCSR() {
		return rowSelectionCSR(ineqRows, ineqSign);
	}

	private CSRMatrix rowSelectionCSR(int[] rows, int[] signs) {
		// If A is dense, fall back to the existing dense->CSR conversion to
		// avoid touching every (i, j) on a sparse build path.
		if (!A.isSparse()) {
			return UjmpBridge.toCSR(rowSelection(rows, signs));
		}
		// Direct CSR build: walk the selected rows, copying non-zeros only.
		int n = rows.length;
		int cols = (int) A.getColumnCount();
		// Two-pass: first count nnz per output row to size data/indices.
		int totalNnz = 0;
		int[] rowNnz = new int[n];
		for (int k = 0; k < n; ++k) {
			int i = rows[k];
			int count = 0;
			for (int j = 0; j < cols; ++j) {
				if (A.getAsDouble(i, j) != 0.0) ++count;
			}
			rowNnz[k] = count;
			totalNnz += count;
		}
		int[] indptr = new int[n + 1];
		int[] indices = new int[totalNnz];
		double[] data = new double[totalNnz];
		int pos = 0;
		for (int k = 0; k < n; ++k) {
			indptr[k] = pos;
			int i = rows[k];
			int sign = signs[k];
			for (int j = 0; j < cols; ++j) {
				double v = A.getAsDouble(i, j);
				if (v != 0.0) {
					indices[pos] = j;
					data[pos] = sign * v;
					++pos;
				}
			}
		}
		indptr[n] = pos;
		return new CSRMatrix(n, cols, indptr, indices, data);
	}
}
