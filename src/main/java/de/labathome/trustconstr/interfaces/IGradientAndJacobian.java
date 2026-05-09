package de.labathome.trustconstr.interfaces;

import de.labathome.trustconstr.records.GradientAndJacobian;
import de.labathome.trustconstr.matrix.Matrix;

/** Combined evaluator returning {@code (gradf(x), gradconstr(x))} in a single call. */
@FunctionalInterface
public interface IGradientAndJacobian {

	/**
	 * @param z current iterate
	 * @return gradient and Jacobian at {@code z}
	 */
	public GradientAndJacobian gradAndJac(Matrix z);
}
