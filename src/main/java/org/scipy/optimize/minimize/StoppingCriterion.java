package org.scipy.optimize.minimize;

import org.ujmp.core.Matrix;

public interface StoppingCriterion {

	public boolean shouldStop(
			State state,
			Matrix x,
			boolean lastIterationFailed,
			double optimality, double constrViolation,
			double trustRadius, double penalty, CGInfo cgInfo);

}
