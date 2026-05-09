package org.scipy.optimize.minimize.records;

import org.scipy.optimize.minimize.matrix.Matrix;

/**
 * Output of {@link org.scipy.optimize.minimize.NumDiff#adjustSchemeToBounds}:
 * the per-dimension finite-difference step (with sign flips and clamping
 * applied so each step lands inside the user-provided bounds) plus a per-
 * dimension flag indicating whether the central scheme had to be downgraded
 * to a one-sided scheme.
 */
public final class AdjustedDifferencingScheme {

	private Matrix hAdjusted;
	private boolean[] useOneSided;

	/**
	 * @param hAdjusted   adjusted absolute step sizes ({@code n x 1})
	 * @param useOneSided per-dimension flag: {@code true} forces a one-sided
	 *                    scheme at this dimension, {@code false} keeps the
	 *                    requested two-sided scheme
	 */
	public AdjustedDifferencingScheme(Matrix hAdjusted, boolean[] useOneSided) {
		this.hAdjusted = hAdjusted;
		this.useOneSided = useOneSided;
	}

	/** @return the bound-adjusted step sizes ({@code n x 1}) */
	public Matrix hAdjusted() {
		return hAdjusted;
	}

	/** @return per-dimension one-sided/two-sided scheme flags */
	public boolean[] useOneSided() {
		return useOneSided;
	}
}
