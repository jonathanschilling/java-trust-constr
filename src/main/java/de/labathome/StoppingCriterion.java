package de.labathome;

import org.ujmp.core.Matrix;

import de.labathome.optimization.PCGResult;

public interface StoppingCriterion {

	public boolean shouldStop(
			State state,
			Matrix x,
			boolean lastIterationFailed,
			double optimality, double constrViolation,
			double trustRadius, double penalty, PCGResult cgInfo);

}
