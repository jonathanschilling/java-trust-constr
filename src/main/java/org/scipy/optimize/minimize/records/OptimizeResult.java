package org.scipy.optimize.minimize.records;

import org.scipy.optimize.minimize.enums.PCGStoppingCondition;
import org.scipy.optimize.minimize.enums.TrustConstrMethod;
import org.scipy.optimize.minimize.matrix.Matrix;

/**
 * Mutable output of
 * {@link org.scipy.optimize.minimize.MinimizeTrustConstr#minimize}. Mirrors
 * scipy's {@code OptimizeResult}: the solution, KKT-related residuals,
 * iteration counters, termination metadata, and constraint multipliers.
 */
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
	 * Whether the optimizer reports success.
	 *
	 * <p>Derived at result construction:
	 * <ul>
	 *   <li>{@link #status} {@code == 1} (gtol satisfied): always {@code true}
	 *       -- gtol-termination requires the constraint violation to also be
	 *       below {@code gtol}.</li>
	 *   <li>{@link #status} {@code == 2} (xtol satisfied / trust-radius
	 *       collapse): {@code true} only if the residual constraint
	 *       violation is below {@code gtol} -- distinguishes "converged at
	 *       an active boundary" from "stuck on an infeasible problem".</li>
	 *   <li>{@link #status} {@code == 0} (maxIter exceeded) or
	 *       {@link #status} {@code == 3} (callback terminated):
	 *       {@code false}.</li>
	 * </ul>
	 *
	 * <p>Mirrors scipy's {@code OptimizeResult.success} semantics, including
	 * the behaviour exercised by scipy {@code test_issue_18882} where a
	 * degenerate constraint causes trust-radius collapse with a non-trivial
	 * constraint violation -- that outcome reports {@code success=false}.
	 */
	public boolean success;

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

	/**
	 * One-line summary suitable for {@code println(result)}: success flag,
	 * status, method, x, fun, optimality, constraint violation, iteration
	 * counters. Mirrors what scipy's {@code repr(OptimizeResult)} shows for
	 * the most-asked-about fields.
	 */
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder("OptimizeResult{");
		sb.append("success=").append(success);
		sb.append(", status=").append(status);
		sb.append(", method=").append(method);
		sb.append(", x=").append(formatVector(x));
		sb.append(", fun=").append(String.format("%.6g", fun));
		sb.append(", optimality=").append(String.format("%.3g", optimality));
		sb.append(", constraintViolation=").append(String.format("%.3g", constraintViolation));
		sb.append(", nIter=").append(nIter);
		sb.append(", numFunctionEval=").append(numFunctionEval);
		if (message != null) {
			sb.append(", message=\"").append(message).append("\"");
		}
		sb.append("}");
		return sb.toString();
	}

	private static String formatVector(Matrix v) {
		if (v == null) return "null";
		long n = v.getRowCount();
		if (n == 0) return "[]";
		StringBuilder sb = new StringBuilder("[");
		long limit = Math.min(n, 6);
		for (long i = 0; i < limit; ++i) {
			if (i > 0) sb.append(", ");
			sb.append(String.format("%.6g", v.getAsDouble(i, 0)));
		}
		if (n > limit) sb.append(", ...(").append(n - limit).append(" more)");
		sb.append("]");
		return sb.toString();
	}
}
