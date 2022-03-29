package org.scipy.optimize.minimize;

import org.scipy.optimize.minimize.interfaces.Constraint;
import org.ujmp.core.Matrix;

public class LinearConstraint implements Constraint {

	@Override
	public Matrix constrEq(Matrix x) {
		return null;
	}

	@Override
	public Matrix constrIneq(Matrix x) {
		return null;
	}

}
