package de.labathome.optimization;

import org.ujmp.core.Matrix;

public class PCGResult {

	/** [n] Solution of the EQP problem. */
	public Matrix x;

	/** Number of iterations. */
	public int niter;

	/**
	 * Reason for algorithm termination:
	 *  1. Iteration limit was reached;
	 *  2. Reached the trust-region boundary;
	 *  3. Negative curvature detected;
	 *  4. Tolerance was satisfied.
	 */
	public int stopCond;

	/** List containing all intermediary vectors (optional). */
	public Matrix[] allVecs;

	/**
	 * True if the proposed step is on the boundary
	 * of the trust region.
	 */
	public boolean hitsBoundary;

}
