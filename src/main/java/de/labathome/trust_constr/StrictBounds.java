package de.labathome.trust_constr;

public class StrictBounds extends FiniteDifferenceBounds {

	public StrictBounds(double[] lb, double[] ub, boolean keepFeasible, long nVars) {
		super(lb, ub, keepFeasible, nVars);
	}
}
