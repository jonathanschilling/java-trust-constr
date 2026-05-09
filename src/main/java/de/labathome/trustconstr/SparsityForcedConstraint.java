package de.labathome.trustconstr;

import de.labathome.trustconstr.interfaces.Constraint;
import de.labathome.trustconstr.interfaces.Jacobian;
import de.labathome.trustconstr.matrix.Matrix;
import de.labathome.trustconstr.matrix.SparseMatrix;

/**
 * Decorator that wraps a {@link Constraint} + {@link Jacobian} source and
 * forces the {@code jacEq} / {@code jacIneq} return values to a specific
 * sparsity. Used by {@link MinimizeTrustConstr#minimizeTrustConstr} to
 * honor the user-supplied {@code sparseJacobian} preference (mirrors
 * scipy's {@code sparse_jacobian=True/False} argument): when explicitly
 * set, callers can override the auto-detected representation.
 *
 * <p>{@code constrEq} / {@code constrIneq} pass through unchanged
 * (constraint values are 1-D and sparsity is rarely meaningful for them).
 * {@code lagrangianContribution}, {@code nEq()}, {@code nIneq()},
 * {@code keepFeasible}-related delegates, and {@code validateKeepFeasibleAtStart}
 * pass through to the source as well.
 */
public final class SparsityForcedConstraint implements Constraint, Jacobian {

	private final Object source;
	private final Constraint sourceConstraint;
	private final Jacobian sourceJacobian;
	/** {@code true} = force sparse output; {@code false} = force dense. */
	private final boolean wantSparse;

	/**
	 * @param source     wrapped constraint (must implement both
	 *                   {@link Constraint} and {@link Jacobian})
	 * @param wantSparse {@code true} forces sparse Jacobian output;
	 *                   {@code false} forces dense
	 */
	public SparsityForcedConstraint(Object source, boolean wantSparse) {
		if (source == null) {
			throw new IllegalArgumentException("source constraint must be non-null");
		}
		this.source = source;
		this.sourceConstraint = (Constraint) source;
		this.sourceJacobian = (Jacobian) source;
		this.wantSparse = wantSparse;
	}

	/** @return number of equality rows in the wrapped source */
	public int nEq() {
		if (source instanceof LinearConstraint) return ((LinearConstraint) source).nEq();
		if (source instanceof NonlinearConstraint) return ((NonlinearConstraint) source).nEq();
		if (source instanceof CombinedConstraint) return ((CombinedConstraint) source).nEq();
		throw new IllegalStateException("Unsupported source type: " + source.getClass());
	}

	/** @return number of canonical inequality rows in the wrapped source */
	public int nIneq() {
		if (source instanceof LinearConstraint) return ((LinearConstraint) source).nIneq();
		if (source instanceof NonlinearConstraint) return ((NonlinearConstraint) source).nIneq();
		if (source instanceof CombinedConstraint) return ((CombinedConstraint) source).nIneq();
		throw new IllegalStateException("Unsupported source type: " + source.getClass());
	}

	@Override
	public Matrix constrEq(Matrix x) {
		return sourceConstraint.constrEq(x);
	}

	@Override
	public Matrix constrIneq(Matrix x) {
		return sourceConstraint.constrIneq(x);
	}

	@Override
	public Matrix jacEq(Matrix x) {
		return forceSparsity(sourceJacobian.jacEq(x));
	}

	@Override
	public Matrix jacIneq(Matrix x) {
		return forceSparsity(sourceJacobian.jacIneq(x));
	}

	/**
	 * Convert {@code m} to the configured sparsity. No-op when already in
	 * the requested representation.
	 *
	 * @param m source matrix
	 * @return either {@code m} unchanged or a fresh matrix in the requested
	 *         representation
	 */
	private Matrix forceSparsity(Matrix m) {
		if (m.isSparse() == wantSparse) return m;
		long rows = m.getRowCount();
		long cols = m.getColumnCount();
		if (wantSparse) {
			SparseMatrix s = SparseMatrix.Factory.zeros(rows, cols);
			for (long r = 0; r < rows; ++r) {
				for (long c = 0; c < cols; ++c) {
					double v = m.getAsDouble(r, c);
					if (v != 0.0) s.setAsDouble(v, r, c);
				}
			}
			return s;
		} else {
			Matrix d = Matrix.Factory.zeros(rows, cols);
			for (long r = 0; r < rows; ++r) {
				for (long c = 0; c < cols; ++c) {
					d.setAsDouble(m.getAsDouble(r, c), r, c);
				}
			}
			return d;
		}
	}

	/**
	 * Pass-through to the source's Lagrangian-Hessian contribution.
	 *
	 * @param x     iterate
	 * @param vEq   equality multipliers
	 * @param vIneq inequality multipliers
	 * @return summed constraint Hessian, or {@code null} if the source has none
	 */
	public Matrix lagrangianContribution(Matrix x, double[] vEq, double[] vIneq) {
		if (source instanceof LinearConstraint) {
			return null;  // linear constraints contribute zero
		}
		if (source instanceof NonlinearConstraint) {
			return ((NonlinearConstraint) source).lagrangianContribution(x, vEq, vIneq);
		}
		if (source instanceof CombinedConstraint) {
			return ((CombinedConstraint) source).lagrangianContribution(x, vEq, vIneq);
		}
		throw new IllegalStateException("Unsupported source type: " + source.getClass());
	}

	/**
	 * Pass-through to the source's enforce-feasibility flags.
	 *
	 * @return per-canonical-row strict-feasibility flags
	 */
	public boolean[] enforceFeasibilityIneq() {
		if (source instanceof LinearConstraint) return ((LinearConstraint) source).enforceFeasibilityIneq();
		if (source instanceof NonlinearConstraint) return ((NonlinearConstraint) source).enforceFeasibilityIneq();
		if (source instanceof CombinedConstraint) return ((CombinedConstraint) source).enforceFeasibilityIneq();
		throw new IllegalStateException("Unsupported source type: " + source.getClass());
	}

	/**
	 * Pass-through to the source's keep-feasible-at-start validation.
	 *
	 * @param x0 starting iterate ({@code n x 1})
	 */
	public void validateKeepFeasibleAtStart(Matrix x0) {
		if (source instanceof LinearConstraint) {
			((LinearConstraint) source).validateKeepFeasibleAtStart(x0);
		} else if (source instanceof NonlinearConstraint) {
			((NonlinearConstraint) source).validateKeepFeasibleAtStart(x0);
		} else if (source instanceof CombinedConstraint) {
			((CombinedConstraint) source).validateKeepFeasibleAtStart(x0);
		}
	}

	/**
	 * @return the wrapped source for type-introspection by the orchestrator
	 */
	public Object source() {
		return source;
	}
}
