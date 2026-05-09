package org.scipy.optimize.minimize.records;

import java.util.Optional;

import org.scipy.optimize.minimize.VectorFunction;
import org.scipy.optimize.minimize.interfaces.Constraint;
import org.scipy.optimize.minimize.matrix.Matrix;

/**
 * Container for a {@link Constraint} that has been prepared for use by the
 * trust-constr orchestrator (i.e. wrapped into a {@link VectorFunction} and
 * tagged with the corresponding {@link Bounds}). Mirrors scipy's
 * {@code PreparedConstraint} record.
 *
 * <p><b>Note:</b> the constructors are currently stubs -- the full preparation
 * logic ({@code _prepare_constraint} in scipy) is on the to-do list inside
 * {@code MinimizeTrustConstr.minimizeTrustConstr}.
 */
public class PreparedConstraint {

	private VectorFunction fun;
	private Bounds bounds;

	/**
	 * Stub constructor that will eventually wrap {@code c} into a
	 * {@link VectorFunction} ready for the orchestrator.
	 *
	 * @param c                user-supplied constraint
	 * @param x0               starting iterate ({@code n x 1})
	 * @param sparseJacobian   user preference for Jacobian representation
	 * @param finiteDiffBounds bounds to clamp FD perturbations within
	 */
	public PreparedConstraint(Constraint c, Matrix x0, Optional<Boolean> sparseJacobian, FiniteDifferenceBounds finiteDiffBounds) {

	}

	/**
	 * Stub constructor for the "promote {@link Bounds} into a constraint"
	 * code path.
	 *
	 * @param bounds         box bounds to fold into the constraint set
	 * @param x0             starting iterate
	 * @param sparseJacobian user preference for Jacobian representation
	 */
	public PreparedConstraint(Bounds bounds, Matrix x0, Optional<Boolean> sparseJacobian) {

	}

	/** @return the {@link Bounds} attached to this prepared constraint */
	public Bounds bounds() {
		return bounds;
	}

	/** @return the wrapped {@link VectorFunction} (constraint + Jacobian) */
	public VectorFunction fun() {
		return fun;
	}
}
