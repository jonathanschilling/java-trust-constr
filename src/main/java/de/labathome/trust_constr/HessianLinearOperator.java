package de.labathome.trust_constr;

public class HessianLinearOperator implements Hessian {

	HessianProduct hessp;
	long nVars;

	public HessianLinearOperator(HessianProduct hessp, long nVars) {
		this.hessp = hessp;
		this.nVars = nVars;
	}

}
