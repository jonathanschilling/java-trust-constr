package org.scipy.optimize.minimize.records;

import org.scipy.optimize.minimize.matrix.Matrix;

public final class AdjustedDifferencingScheme {

	private Matrix hAdjusted;
	private boolean[] useOneSided;

	public AdjustedDifferencingScheme(Matrix hAdjusted, boolean[] useOneSided) {
		this.hAdjusted = hAdjusted;
		this.useOneSided = useOneSided;
	}

	public Matrix hAdjusted() {
		return hAdjusted;
	}

	public boolean[] useOneSided() {
		return useOneSided;
	}

}
