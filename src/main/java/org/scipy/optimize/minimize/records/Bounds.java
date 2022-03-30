package org.scipy.optimize.minimize.records;

public final class Bounds {

	private double[] lb;
	private double[] ub;
	private boolean keepFeasible;

	public Bounds(double[] lb, double[] ub, boolean keepFeasible) {
		this.lb = lb;
		this.ub = ub;
		this.keepFeasible = keepFeasible;
	}

	public double[] lb() {
		return lb;
	}

	public double[] ub() {
		return ub;
	}

	public boolean keepFeasible() {
		return keepFeasible;
	}
}
