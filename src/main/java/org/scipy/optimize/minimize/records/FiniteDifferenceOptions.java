package org.scipy.optimize.minimize.records;

import org.scipy.optimize.minimize.FiniteDifferenceBounds;
import org.scipy.optimize.minimize.enums.FiniteDifferenceMethod;

public final class FiniteDifferenceOptions {

	private FiniteDifferenceMethod method;
	private double relStep;
	private double[] absStep;
	private FiniteDifferenceBounds bounds;
	private boolean asLinearOperator;
	private Sparsity sparsity;

	public FiniteDifferenceOptions(FiniteDifferenceMethod method, double relStep, double[] absStep,
			FiniteDifferenceBounds bounds, boolean asLinearOperator, Sparsity sparsity) {
		this.method = method;
		this.relStep = relStep;
		this.absStep = absStep;
		this.bounds = bounds;
		this.asLinearOperator = asLinearOperator;
		this.sparsity = sparsity;
	}

	public FiniteDifferenceMethod method() {
		return method;
	}

	public double relStep() {
		return relStep;
	}

	public double[] absStep() {
		return absStep;
	}

	public FiniteDifferenceBounds bounds() {
		return bounds;
	}

	public boolean asLinearOperator() {
		return asLinearOperator;
	}

	public Sparsity sparsity() {
		return sparsity;
	}
}
