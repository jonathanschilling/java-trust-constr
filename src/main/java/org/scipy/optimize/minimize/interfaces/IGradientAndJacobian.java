package org.scipy.optimize.minimize.interfaces;

import org.scipy.optimize.minimize.records.GradientAndJacobian;
import org.scipy.optimize.minimize.matrix.Matrix;

/** Combined evaluator returning {@code (gradf(x), gradconstr(x))} in a single call. */
@FunctionalInterface
public interface IGradientAndJacobian {

	/**
	 * @param z current iterate
	 * @return gradient and Jacobian at {@code z}
	 */
	public GradientAndJacobian gradAndJac(Matrix z);
}
