package de.labathome.trustconstr.records;

import de.labathome.trustconstr.enums.PCGStoppingCondition;
import de.labathome.trustconstr.matrix.Matrix;

/**
 * Mutable state threaded through the equality-path SQP outer loop. Updated
 * once per iteration by {@link de.labathome.trustconstr.MinimizeTrustConstr#updateState}
 * and consulted by the user-supplied {@link
 * de.labathome.trustconstr.interfaces.IterationCallback}.
 */
public class State {

	/** Number of completed outer iterations. */
	public int nIter;
	/** Number of objective evaluations so far. */
	public int numEval;
	/** Number of gradient evaluations so far. */
	public int numGradientEval;
	/** Number of Hessian evaluations so far. */
	public int numHessianEval;

	/** Per-constraint count of constraint evaluations. */
	public int[] numConstraintEval;
	/** Per-constraint count of constraint Jacobian evaluations. */
	public int[] numConstraintJacobianEval;
	/** Per-constraint count of constraint Hessian evaluations. */
	public int[] numConstraintHessianEval;

	/** Wall-clock time elapsed since the start of the run, in nanoseconds. */
	public long executionTime;

	/** Current trust-region radius. */
	public double trustRadius;
	/** Current merit-function penalty (the {@code rho} in {@code f(x) + rho ||c(x)||}). */
	public double constraintPenalty;

	/** Number of CG iterations performed in the most recent inner solve. */
	public int cgNIter;
	/** Why the most recent CG inner solve terminated. */
	public PCGStoppingCondition cgStopCond;

	/** Current iterate ({@code n x 1}). */
	public Matrix x;
	/** Objective value {@code f(x)} at the current iterate. */
	public double fun;
	/** Gradient {@code gradf(x)} at the current iterate. */
	public Matrix grad;

	/** Lagrange multipliers, one entry per constraint source. */
	public Matrix[] v;
	/** Constraint residual vectors, one entry per constraint source. */
	public Matrix[] constr;
	/** Constraint Jacobians, one entry per constraint source. */
	public Matrix[] jac;

	/** Lagrangian gradient {@code gradf(x) + Sum v_i gradc_i(x)} at the current iterate. */
	public Matrix lagrangianGrad;

	/** KKT optimality measure (infinity-norm of the Lagrangian gradient). */
	public double optimality;
	/** Infinity-norm of the constraint residual at the current iterate. */
	public double constrViolation;

	/** Default-construct a {@link State} with all fields at their zero/null defaults. */
	public State() {}
}
