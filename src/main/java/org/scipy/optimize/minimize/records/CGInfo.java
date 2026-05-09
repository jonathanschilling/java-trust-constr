package org.scipy.optimize.minimize.records;

import java.util.List;

import org.scipy.optimize.minimize.enums.PCGStoppingCondition;
import org.scipy.optimize.minimize.matrix.Matrix;

public class CGInfo {

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
