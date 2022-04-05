package org.scipy.optimize.minimize.records;

import org.ujmp.core.Matrix;

public final class StatefulResult {

	private Matrix x;
	private State state;

	public StatefulResult(Matrix x, State state) {
		this.x = x;
		this.state = state;
	}

	public Matrix x() {
		return x;
	}

	public State state() {
		return state;
	}
}
