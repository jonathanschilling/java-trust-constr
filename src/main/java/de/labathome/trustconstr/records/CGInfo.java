package de.labathome.trustconstr.records;

import java.util.List;

import de.labathome.trustconstr.enums.PCGStoppingCondition;
import de.labathome.trustconstr.matrix.Matrix;

/**
 * Result of a projected-CG inner solve from
 * {@link de.labathome.trustconstr.QPSubproblem#projectedCG}. Carries the
 * solution vector, iteration count, why CG terminated, optionally the
 * full iterate trajectory, and a flag indicating whether the proposed step
 * landed on the trust-region boundary.
 */
public class CGInfo {

	/** Solution vector of the EQP problem ({@code n x 1}). */
	public Matrix x;

	/** Number of CG iterations performed. */
	public int niter;

	/** Why CG terminated (residual met tolerance, hit boundary, etc.). */
	public PCGStoppingCondition stopCond;

	/** All intermediate iterates, when {@code returnAll=true}; otherwise {@code null}. */
	public List<Matrix> allVecs;

	/** {@code true} iff the returned step lies on the trust-region boundary. */
	public boolean hitsBoundary;
}
