package de.labathome.trust_constr;

import java.util.Arrays;

public class FiniteDifferenceBounds {

	double[] lb;
	double[] ub;
	boolean keepFeasible;
	long nVars;

	public FiniteDifferenceBounds(double[] lb, double[] ub, boolean keepFeasible, long nVars) {
		this.lb = lb;
		this.ub = ub;
		this.keepFeasible = keepFeasible;
		this.nVars = nVars;
	}

	public static FiniteDifferenceBounds unbounded(long nVars) {

		double[] lb = new double[(int) nVars];
		Arrays.fill(lb, Double.NEGATIVE_INFINITY);

		double[] ub = new double[(int) nVars];
		Arrays.fill(ub, Double.POSITIVE_INFINITY);

		return new FiniteDifferenceBounds(lb, ub, false, nVars);
	}
}
