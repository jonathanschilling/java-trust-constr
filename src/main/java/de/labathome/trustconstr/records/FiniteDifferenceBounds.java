package de.labathome.trustconstr.records;

import de.labathome.trustconstr.matrix.Matrix;

/**
 * Box bounds {@code lb &le; x &le; ub} used by
 * {@link de.labathome.trustconstr.NumDiff#approxDerivative} to clamp
 * finite-difference perturbations.
 */
public class FiniteDifferenceBounds {

	private Matrix lb;
	private Matrix ub;
	private boolean keepFeasible;

	/**
	 * @param lb           per-variable lower bound ({@code n &times; 1})
	 * @param ub           per-variable upper bound ({@code n &times; 1})
	 * @param keepFeasible if {@code true}, every FD perturbation must stay
	 *                     strictly inside the box
	 */
	public FiniteDifferenceBounds(Matrix lb, Matrix ub, boolean keepFeasible) {
		this.lb = lb;
		this.ub = ub;
		this.keepFeasible = keepFeasible;
	}

	/**
	 * Convenience factory for the bound-free case: {@code -inf &le; x &le; +inf}.
	 *
	 * @param nVars number of decision variables
	 * @return an unbounded {@link FiniteDifferenceBounds} of the requested size
	 */
	public static FiniteDifferenceBounds unbounded(long nVars) {
		Matrix lb = Matrix.Factory.ones(nVars, 1).times(Double.NEGATIVE_INFINITY);
		Matrix ub = Matrix.Factory.ones(nVars, 1).times(Double.POSITIVE_INFINITY);
		return new FiniteDifferenceBounds(lb, ub, false);
	}

	/** @return lower bounds ({@code n &times; 1}) */
	public Matrix lb() {
		return lb;
	}

	/** @return upper bounds ({@code n &times; 1}) */
	public Matrix ub() {
		return ub;
	}

	/** @return whether perturbations must stay strictly inside the box */
	public boolean keepFeasible() {
		return keepFeasible;
	}
}
