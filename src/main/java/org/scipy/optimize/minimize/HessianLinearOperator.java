package org.scipy.optimize.minimize;

import java.util.function.BiFunction;

import org.ujmp.core.Matrix;

public class HessianLinearOperator implements BiFunction<Matrix, Object, Matrix> {

	HessianProduct hessp;
	long nVars;

	public HessianLinearOperator(HessianProduct hessp, long nVars) {
		this.hessp = hessp;
		this.nVars = nVars;
	}

	@Override
	public Matrix apply(Matrix t, Object args) {
		// TODO Auto-generated method stub
		return null;
	}

}
