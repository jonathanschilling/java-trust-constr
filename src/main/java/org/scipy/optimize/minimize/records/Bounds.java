package org.scipy.optimize.minimize.records;

import org.ujmp.core.Matrix;

public final class Bounds {

	private Matrix lb;
	private Matrix ub;
	private boolean keepFeasible;

	public Bounds(Matrix lb, Matrix ub, boolean keepFeasible) {
		this.lb = lb;
		this.ub = ub;
		this.keepFeasible = keepFeasible;
	}

	public Matrix lb() {
		return lb;
	}

	public Matrix ub() {
		return ub;
	}

	public boolean keepFeasible() {
		return keepFeasible;
	}
}
