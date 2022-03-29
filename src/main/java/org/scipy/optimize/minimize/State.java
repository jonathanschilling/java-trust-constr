package org.scipy.optimize.minimize;

import org.scipy.optimize.minimize.enums.PCGStoppingCondition;
import org.ujmp.core.Matrix;

public class State {

	int nIter;
	int numEval;
	int numGradientEval;
	int numHessianEval;

	int[] numConstraintEval;
	int[] numConstraintJacobianEval;
	int[] numConstraintHessianEval;

	long executionTime;

	double trustRadius;
	double constraintPenalty;

	int cgNIter;
	PCGStoppingCondition cgStopCond;

	Matrix x;
	double fun;
	Matrix grad;

	Matrix[] v;
	Matrix[] constr;
	Matrix[] jac;

	Matrix lagrangianGrad;

	double optimality;
	double constrViolation;
}
