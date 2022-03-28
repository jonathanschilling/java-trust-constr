package de.labathome.trust_constr;

import org.ujmp.core.Matrix;

public interface GlobalStoppingCriteria {

	public boolean shouldStop(
			State state,
			Matrix x,
			boolean lastIterationFailed,
			double optimality, double constrViolation,
			double trustRadius, double penalty, PCGResult cgInfo,
			double barrierParameter, double tolerance);
}
