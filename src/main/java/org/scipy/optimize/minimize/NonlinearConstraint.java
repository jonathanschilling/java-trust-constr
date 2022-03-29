package org.scipy.optimize.minimize;

import org.ujmp.core.Matrix;

public class NonlinearConstraint implements Constraint {

	@Override
	public Matrix constrEq(Matrix x) {
		return null;
	}

	@Override
	public Matrix constrIneq(Matrix x) {
		return null;
	}

}
