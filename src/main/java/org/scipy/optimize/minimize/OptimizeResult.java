package org.scipy.optimize.minimize;

import org.ujmp.core.Matrix;

public class OptimizeResult {

	/** [n] Solution found. */
	Matrix x;

	/** Infinity norm of the Lagrangian gradient at the solution. */
	double optimality;

	/** Maximum constraint violation at the solution. */
	double constraintViolation;

	/** Objective function at the solution. */
	double fun;

	/** [n] Gradient of the objective function at the solution. */
	Matrix grad;

	/** [n] Gradient of the Lagrangian function at the solution. */
	Matrix lagrangianGrad;

	/** Total number of iterations. */
	int nIter;

	/** Number of the objective function evaluations. */
	int numFunctionEval;

	/** Number of the objective function gradient evaluations. */
	int numJacobianEval;

	/** Number of the objective function Hessian evaluations. */
	int numHessianEval;

	/** Total number of the conjugate gradient method iterations. */
	int cgIter;

	/** Optimization method used. */
	TrustConstrMethod method;

	/** List of constraint values at the solution. */
	Matrix[] constr;

	/** List of the Jacobian matrices of the constraints at the solution. */
	Matrix[] jac;

	/**
	 * List of the Lagrange multipliers for the constraints at the solution. For an
	 * inequality constraint a positive multiplier means that the upper bound is
	 * active, a negative multiplier means that the lower bound is active and if a
	 * multiplier is zero it means the constraint is not active.
	 */
	Matrix[] v;

	/** Number of constraint evaluations for each of the constraints. */
	int[] numConstraintEval;

	/** Number of Jacobian matrix evaluations for each of the constraints. */
	int[] numConstraintJacobianEval;

	/** Number of Hessian evaluations for each of the constraints. */
	int[] numConstraintHessianEval;

	/** Radius of the trust region at the last iteration. */
	double trustRadius;

	/** Penalty parameter at the last iteration, see `initial_constr_penalty`. */
	double constraintPenalty;

	/**
	 * Tolerance for the barrier subproblem at the last iteration.
	 * Only for problems with inequality constraints.
	 */
	double barrierTolerance;

	/**
	 * Barrier parameter at the last iteration.
	 * Only for problems with inequality constraints.
	 */
	double barrierParameter;

	/** Total execution time in nanoseconds. */
	long executionTime;

	/** Termination message. */
	String message;

	/**
	 * {0, 1, 2, 3}: Termination status:
	 *
	 * 0 : The maximum number of function evaluations is exceeded.
	 * 1 : `gtol` termination condition is satisfied.
	 * 2 : `xtol` termination condition is satisfied.
	 * 3 : `callback` function requested termination.
	 */
	int status;

	/**
	 * Reason for CG subproblem termination at the last iteration:
	 *
	 * 0 : CG subproblem not evaluated.
	 * 1 : Iteration limit was reached.
	 * 2 : Reached the trust-region boundary.
	 * 3 : Negative curvature detected.
	 * 4 : Tolerance was satisfied.
	 */
	PCGStoppingCondition cgStopCond;

}
