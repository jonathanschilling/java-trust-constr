package org.scipy.optimize.minimize;

public class FiniteDifferenceOptions {

	public FiniteDifferenceMethod method;
	public double relStep;
	public double[] absStep;
	public FiniteDifferenceBounds bounds;
	public boolean asLinearOperator;

	public FiniteDifferenceOptions() {
		asLinearOperator = false;
	}

}
