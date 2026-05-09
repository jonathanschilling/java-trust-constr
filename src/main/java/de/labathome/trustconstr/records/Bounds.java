package de.labathome.trustconstr.records;

import de.labathome.trustconstr.matrix.Matrix;

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

	/**
	 * Lower and upper residuals at {@code x}: {@code sl = x - lb} (positive
	 * iff lower bound satisfied), {@code sb = ub - x} (positive iff upper
	 * bound satisfied). Mirrors scipy's {@code Bounds.residual(x)}.
	 *
	 * @param x current iterate ({@code n x 1})
	 * @return {@code (sl, sb)} as a {@link Residual}
	 */
	public Residual residual(Matrix x) {
		long n = lb.getRowCount();
		double[] sl = new double[(int) n];
		double[] sb = new double[(int) n];
		for (int i = 0; i < n; ++i) {
			double xi = x.getAsDouble(i, 0);
			sl[i] = xi - lb.getAsDouble(i, 0);
			sb[i] = ub.getAsDouble(i, 0) - xi;
		}
		return new Residual(sl, sb);
	}
}
