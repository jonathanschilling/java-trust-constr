package de.labathome.optimization;

import java.util.List;

import org.ujmp.core.Matrix;

public class PCGResult {

	public enum PCGStoppingCondition {
		/** 1: Iteration limit was reached */
		ITER_LIMIT_REACHED,

		/** 2: Reached the trust-region boundary */
		TRUST_REGION_BOUNDARY_REACHED,

		/** 3: Negative curvature detected */
		NEGATIVE_CURVATURE,

		/** 4: Tolerance was satisfied */
		TOLERANCE_SATISFIED
	}

	/** [n] Solution of the EQP problem. */
	public Matrix x;

	/** Number of iterations. */
	public int niter;

	public PCGStoppingCondition stopCond;

	/** List containing all intermediary vectors (optional). */
	public List<Matrix> allVecs;

	/**
	 * True if the proposed step is on the boundary
	 * of the trust region.
	 */
	public boolean hitsBoundary;
}
