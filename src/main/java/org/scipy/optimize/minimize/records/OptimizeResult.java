package org.scipy.optimize.minimize.records;

import org.scipy.optimize.minimize.enums.PCGStoppingCondition;
import org.scipy.optimize.minimize.enums.TrustConstrMethod;
import org.ujmp.core.Matrix;

public class OptimizeResult {

	/** [n] Solution found. */
	public Matrix x;

	/** Infinity norm of the Lagrangian gradient at the solution. */
	public double optimality;

	/** Maximum constraint violation at the solution. */
	public double constraintViolation;

	/** Objective function at the solution. */
	public double fun;

	/** [n] Gradient of the objective function at the solution. */
	public Matrix grad;

	/** [n] Gradient of the Lagrangian function at the solution. */
	public Matrix lagrangianGrad;

	/** Total number of iterations. */
	public int nIter;

	/** Number of the objective function evaluations. */
	public int numFunctionEval;

	/** Number of the objective function gradient evaluations. */
	public int numJacobianEval;

	/** Number of the objective function Hessian evaluations. */
	public int numHessianEval;

	/** Total number of the conjugate gradient method iterations. */
	public int cgIter;

	/** Optimization method used. */
	public TrustConstrMethod method;

	/** List of constraint values at the solution. */
	public Matrix[] constr;

	/** List of the Jacobian matrices of the constraints at the solution. */
	public Matrix[] jac;

	/**
	 * List of the Lagrange multipliers for the constraints at the solution. For an
	 * inequality constraint a positive multiplier means that the upper bound is
	 * active, a negative multiplier means that the lower bound is active and if a
	 * multiplier is zero it means the constraint is not active.
	 */
	public Matrix[] v;

	/** Number of constraint evaluations for each of the constraints. */
	public int[] numConstraintEval;

	/** Number of Jacobian matrix evaluations for each of the constraints. */
	public int[] numConstraintJacobianEval;

	/** Number of Hessian evaluations for each of the constraints. */
	public int[] numConstraintHessianEval;

	/** Radius of the trust region at the last iteration. */
	public double trustRadius;

	/** Penalty parameter at the last iteration, see `initial_constr_penalty`. */
	public double constraintPenalty;

	/**
	 * Tolerance for the barrier subproblem at the last iteration.
	 * Only for problems with inequality constraints.
	 */
	public double barrierTolerance;

	/**
	 * Barrier parameter at the last iteration.
	 * Only for problems with inequality constraints.
	 */
	public double barrierParameter;

	/** Total execution time in nanoseconds. */
	public long executionTime;

	/** Termination message. */
	public String message;

	/**
	 * {0, 1, 2, 3}: Termination status:
	 *
	 * 0 : The maximum number of function evaluations is exceeded.
	 * 1 : `gtol` termination condition is satisfied.
	 * 2 : `xtol` termination condition is satisfied.
	 * 3 : `callback` function requested termination.
	 */
	public int status;

	/**
	 * Reason for CG subproblem termination at the last iteration:
	 *
	 * 0 : CG subproblem not evaluated.
	 * 1 : Iteration limit was reached.
	 * 2 : Reached the trust-region boundary.
	 * 3 : Negative curvature detected.
	 * 4 : Tolerance was satisfied.
	 */
	public PCGStoppingCondition cgStopCond;

}
