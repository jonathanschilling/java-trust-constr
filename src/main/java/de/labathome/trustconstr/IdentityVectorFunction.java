package de.labathome.trustconstr;

import java.util.Optional;

import de.labathome.trustconstr.matrix.Matrix;
import de.labathome.trustconstr.matrix.SparseMatrix;

/**
 * Identity vector function {@code f(x) = x} (Jacobian {@code I}). Used by
 * {@link de.labathome.trustconstr.records.PreparedConstraint} when a
 * {@link de.labathome.trustconstr.records.Bounds} object is folded into
 * the constraint set.
 *
 * <p>Counterpart to scipy's {@code IdentityVectorFunction}
 * ({@code scipy/optimize/_differentiable_functions.py:829}).
 */
public class IdentityVectorFunction extends LinearVectorFunction {

	/**
	 * Build an identity vector function sized for the iterate {@code x0}.
	 *
	 * @param x0             starting iterate ({@code n x 1})
	 * @param sparseJacobian {@code Optional.of(true)} or
	 *                       {@code Optional.empty()} stores {@code I} sparse;
	 *                       {@code Optional.of(false)} stores it dense
	 */
	public IdentityVectorFunction(Matrix x0, Optional<Boolean> sparseJacobian) {
		super(buildIdentity(x0.getRowCount(),
				sparseJacobian.isEmpty() || sparseJacobian.get()),
				x0,
				Optional.of(sparseJacobian.isEmpty() || sparseJacobian.get()));
	}

	private static Matrix buildIdentity(long n, boolean sparse) {
		if (sparse) {
			SparseMatrix m = SparseMatrix.Factory.zeros(n, n);
			for (long i = 0; i < n; ++i) {
				m.setAsDouble(1.0, i, i);
			}
			return m;
		}
		return Matrix.Factory.eye(n, n);
	}
}
