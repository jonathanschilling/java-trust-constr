package org.scipy.optimize.minimize.records;

import org.scipy.optimize.minimize.matrix.Matrix;

/**
 * Box bounds {@code lb &le; x &le; ub} on the optimisation variable, plus
 * a global {@code keepFeasible} flag controlling whether the algorithm is
 * allowed to step outside the box during line-searches and trial steps.
 */
public final class Bounds {

	private Matrix lb;
	private Matrix ub;
	private boolean keepFeasible;

	/**
	 * @param lb           per-variable lower bound ({@code n x 1});
	 *                     use {@link Double#NEGATIVE_INFINITY} to disable a bound
	 * @param ub           per-variable upper bound ({@code n x 1});
	 *                     use {@link Double#POSITIVE_INFINITY} to disable a bound
	 * @param keepFeasible if {@code true}, every trial iterate must stay
	 *                     within the bounds
	 */
	public Bounds(Matrix lb, Matrix ub, boolean keepFeasible) {
		this.lb = lb;
		this.ub = ub;
		this.keepFeasible = keepFeasible;
	}

	/** @return the lower bound ({@code n x 1}) */
	public Matrix lb() {
		return lb;
	}

	/** @return the upper bound ({@code n x 1}) */
	public Matrix ub() {
		return ub;
	}

	/** @return whether trial iterates must stay strictly inside the box */
	public boolean keepFeasible() {
		return keepFeasible;
	}
}
