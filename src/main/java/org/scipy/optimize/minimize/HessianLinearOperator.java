package org.scipy.optimize.minimize;

import org.scipy.optimize.minimize.interfaces.Hessian;
import org.ujmp.core.Matrix;

public class HessianLinearOperator implements Hessian {

	HessianProduct hessp;
	long nVars;

	public HessianLinearOperator(HessianProduct hessp, long nVars) {
		this.hessp = hessp;
		this.nVars = nVars;
	}

	@Override
	public Matrix hess(Matrix x, Object args) {
		// TODO Auto-generated method stub
		return null;
	}

}
