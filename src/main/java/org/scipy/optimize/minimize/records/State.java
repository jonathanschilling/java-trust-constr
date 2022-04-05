package org.scipy.optimize.minimize.records;

import org.scipy.optimize.minimize.enums.PCGStoppingCondition;
import org.ujmp.core.Matrix;

public class State {

	public int nIter;
	public int numEval;
	public int numGradientEval;
	public int numHessianEval;

	public int[] numConstraintEval;
	public int[] numConstraintJacobianEval;
	public int[] numConstraintHessianEval;

	public long executionTime;

	public double trustRadius;
	public double constraintPenalty;

	public int cgNIter;
	public PCGStoppingCondition cgStopCond;

	public Matrix x;
	public double fun;
	public Matrix grad;

	public Matrix[] v;
	public Matrix[] constr;
	public Matrix[] jac;

	public Matrix lagrangianGrad;

	public double optimality;
	public double constrViolation;
}
