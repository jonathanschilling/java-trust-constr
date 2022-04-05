package org.scipy.optimize.minimize.interfaces;

import org.scipy.optimize.minimize.records.CGInfo;
import org.scipy.optimize.minimize.records.State;
import org.ujmp.core.Matrix;

public interface GlobalStoppingCriteria {

	public boolean shouldStop(State state, Matrix x, boolean lastIterationFailed, double optimality,
			double constrViolation, double trustRadius, double penalty, CGInfo cgInfo, double barrierParameter,
			double tolerance);
}
