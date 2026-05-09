package org.scipy.optimize.minimize;

import org.scipy.optimize.minimize.enums.HessianApproximationType;
import org.scipy.optimize.minimize.interfaces.HessianUpdateStrategy;

import java.util.LinkedList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.ToDoubleBiFunction;
import java.util.function.ToDoubleFunction;

import org.scipy.optimize.minimize.enums.FiniteDifferenceMethod;
import org.scipy.optimize.minimize.enums.PCGStoppingCondition;
import org.scipy.optimize.minimize.enums.ProjectionMethod;
import org.scipy.optimize.minimize.enums.TrustConstrMethod;
import org.scipy.optimize.minimize.interfaces.Constraint;
import org.scipy.optimize.minimize.interfaces.HessianProduct;
import org.scipy.optimize.minimize.interfaces.GlobalStoppingCriteria;
import org.scipy.optimize.minimize.interfaces.IFunctionAndConstraint;
import org.scipy.optimize.minimize.interfaces.IGradientAndJacobian;
import org.scipy.optimize.minimize.interfaces.IterationCallback;
import org.scipy.optimize.minimize.interfaces.Jacobian;
import org.scipy.optimize.minimize.interfaces.LagrangeHessian;
import org.scipy.optimize.minimize.interfaces.LinearOperator;
import org.scipy.optimize.minimize.interfaces.StoppingCriterion;
import org.scipy.optimize.minimize.interfaces.VectorFunctionLike;
import org.scipy.optimize.minimize.records.Bounds;
import org.scipy.optimize.minimize.records.CGInfo;
import org.scipy.optimize.minimize.records.FiniteDifferenceBounds;
import org.scipy.optimize.minimize.records.FiniteDifferenceOptions;
import org.scipy.optimize.minimize.records.FunctionAndConstraint;
import org.scipy.optimize.minimize.records.GradientAndJacobian;
import org.scipy.optimize.minimize.records.OptimizeResult;
import org.scipy.optimize.minimize.records.PreparedConstraint;
import org.scipy.optimize.minimize.records.State;
import org.scipy.optimize.minimize.records.StateIP;
import org.scipy.optimize.minimize.records.StatefulResult;
import org.scipy.optimize.minimize.records.StrictBounds;
import org.scipy.optimize.minimize.report.IPReport;
import org.scipy.optimize.minimize.report.SQPReport;
import org.scipy.optimize.minimize.matrix.DenseMatrix;
import org.scipy.optimize.minimize.matrix.LinAlg;
import org.scipy.optimize.minimize.matrix.Matrix;

/** Java port of scipy.optimize.minimize(method='trust-constr') */
public class MinimizeTrustConstr {

	/**
	 * Refresh the equality-path {@link State} record after an outer iteration
	 * (the function value, gradient, optimality, constraint violation,
	 * trust-region and CG fields).
	 *
	 * @param state                outer-iteration state to update
	 * @param x                    current iterate
	 * @param lastIterationFailed  whether the previous step was rejected
	 * @param objective            wrapped objective providing eval-count metadata
	 * @param preparedConstraints  prepared constraint records
	 * @param startTime            wall-clock {@link System#nanoTime()} at the start of the run
	 * @param trustRadius          current trust-region radius
	 * @param constraintPenalty    current merit-function penalty
	 * @param cgInfo               info from the projected-CG inner solve
	 * @return the same {@code state} after the in-place mutation
	 */
	public static State updateState(State state, Matrix x, boolean lastIterationFailed, ScalarFunction objective,
			PreparedConstraint[] preparedConstraints, long startTime, double trustRadius, double constraintPenalty,
			CGInfo cgInfo) {

		state.nIter++;

		state.numEval = objective.numFunctionEvals();
		state.numGradientEval = objective.numGradientEvals();
		state.numHessianEval = objective.numHessianEvals();

		int n = preparedConstraints.length;
		if (state.numConstraintEval == null || state.numConstraintEval.length != n) {
			state.numConstraintEval = new int[n];
			state.numConstraintJacobianEval = new int[n];
			state.numConstraintHessianEval = new int[n];

			state.v = new Matrix[n];
			state.constr = new Matrix[n];
			state.jac = new Matrix[n];
		}
		for (int i = 0; i < preparedConstraints.length; ++i) {
			VectorFunctionLike c = preparedConstraints[i].fun();
			state.numConstraintEval[i] = c.numFunctionEvals();
			state.numConstraintJacobianEval[i] = c.numJacobianEvals();
			state.numConstraintHessianEval[i] = c.numHessianEvals();
		}

		if (!lastIterationFailed) {
			state.x = x;
			state.fun = objective.f();
			state.grad = objective.g();

			for (int i = 0; i < preparedConstraints.length; ++i) {
				VectorFunctionLike c = preparedConstraints[i].fun();
				state.v[i] = c.v();
				state.constr[i] = c.f();
				state.jac[i] = c.J();
			}

			// Compute Lagrangian Gradient
			state.lagrangianGrad = Matrix.Factory.copyFromMatrix(state.grad);
			for (PreparedConstraint c : preparedConstraints) {
				state.lagrangianGrad = state.lagrangianGrad
						.plus(c.fun().J().transpose().mtimes(c.fun().v()));
			}
			state.optimality = state.lagrangianGrad.normInf();

			// Compute maximum constraint violation
			state.constrViolation = 0.0;
			for (int i = 0; i < preparedConstraints.length; ++i) {
				Matrix lb = preparedConstraints[i].bounds().lb();
				Matrix ub = preparedConstraints[i].bounds().ub();
				Matrix c = state.constr[i];
				for (int j = 0; j < lb.getRowCount(); ++j) {
					double lowerViolation = lb.getAsDouble(j, 0) - c.getAsDouble(j, 0);
					double upperViolation = c.getAsDouble(j, 0) - ub.getAsDouble(j, 0);
					double maxCViol = Math.max(lowerViolation, upperViolation);
					state.constrViolation = Math.max(state.constrViolation, maxCViol);
				}
			}
		}

		state.executionTime = System.nanoTime() - startTime;

		state.trustRadius = trustRadius;
		state.constraintPenalty = constraintPenalty;

		state.cgNIter += cgInfo.niter;
		state.cgStopCond = cgInfo.stopCond;

		return state;
	}

	/**
	 * Refresh the IP-path {@link StateIP} record after an outer iteration:
	 * delegates to {@link #updateState} and then sets the barrier-specific
	 * fields.
	 *
	 * @param state                outer-iteration IP state to update
	 * @param x                    current iterate
	 * @param lastIterationFailed  whether the previous step was rejected
	 * @param objective            wrapped objective providing eval-count metadata
	 * @param preparedConstraints  prepared constraint records
	 * @param startTime            wall-clock {@link System#nanoTime()} at the start of the run
	 * @param trustRadius          current trust-region radius
	 * @param constraintPenalty    current merit-function penalty
	 * @param cgInfo               info from the projected-CG inner solve
	 * @param barrierParameter     current log-barrier coefficient
	 * @param barrierTolerance     current barrier-subproblem tolerance
	 * @return the same {@code state} after the in-place mutation
	 */
	public static StateIP updateStateIP(StateIP state, Matrix x, boolean lastIterationFailed, ScalarFunction objective,
			PreparedConstraint[] preparedConstraints, long startTime, double trustRadius, double constraintPenalty,
			CGInfo cgInfo, double barrierParameter, double barrierTolerance) {

		updateState(state, x, lastIterationFailed, objective, preparedConstraints, startTime, trustRadius,
				constraintPenalty, cgInfo);

		state.barrierParameter = barrierParameter;
		state.barrierTolerance = barrierTolerance;

		return state;
	}

	/**
	 * Convenience entry point covering the equality-constrained slice of the
	 * trust-constr API: a scalar objective with an explicit gradient and Hessian,
	 * subject to optional pure-equality constraints from a single
	 * {@link LinearConstraint} or {@link NonlinearConstraint}.
	 *
	 * <p>This is a focused subset of {@code scipy.optimize.minimize(method='trust-constr')}
	 * intended for problems where every constraint row is an equality
	 * ({@code lb[i] == ub[i]}). It dispatches to {@link EqualityConstrainedSQP#eqSQP}
	 * directly without the full canonical-form constraint pipeline. Callers with
	 * inequalities, {@link Bounds}, multiple constraints, or finite-difference
	 * Hessians/Jacobians should wait for the full {@link #minimizeTrustConstr}
	 * orchestrator.
	 *
	 * @param fun objective function {@code f : R^n -> R}
	 * @param grad analytic gradient {@code g : R^n -> R^n}
	 * @param hess analytic Hessian {@code H : R^n -> R^{n x n}}
	 * @param x0   initial point as a column vector (n x 1)
	 * @param eq   equality constraint, or {@code null} for unconstrained problems
	 *             (the constraint object must satisfy {@code nIneq() == 0})
	 * @param maxIter maximum number of outer iterations
	 * @param xtol stop when {@code trustRadius < xtol}
	 * @param gtol stop when {@code optimality < gtol} and {@code constrViolation < gtol}
	 * @param <C>  constraint type, must implement both {@link Constraint} and {@link Jacobian}
	 * @return populated {@link OptimizeResult}
	 */
	public static <C extends Constraint & Jacobian> OptimizeResult minimizeEqualityConstrained(
			Function<Matrix, Double> fun,
			Function<Matrix, Matrix> grad,
			Function<Matrix, Matrix> hess,
			Matrix x0, C eq,
			int maxIter, double xtol, double gtol) {
		if (nIneqOf(eq) != 0) {
			throw new UnsupportedOperationException(
					"Inequality constraints not supported by minimizeEqualityConstrained");
		}
		return minimizeEqualityConstrainedRaw(fun, grad, hess, x0, eq, nEqOf(eq),
				maxIter, xtol, gtol, null, 1.0, 1.0, null);
	}

	private static OptimizeResult minimizeEqualityConstrainedRaw(
			Function<Matrix, Double> fun,
			Function<Matrix, Matrix> grad,
			Function<Matrix, Matrix> hess,
			Matrix x0, Object eq, int nEq,
			int maxIter, double xtol, double gtol,
			IterationCallback callback,
			double initialPenalty,
			double initialTrustRadius,
			ProjectionMethod factorizationMethod) {
		final int nVars = (int) x0.getRowCount();
		final Constraint eqConstr = (Constraint) eq;
		final Jacobian eqJac = (Jacobian) eq;

		// Wrap objective in ScalarFunction for the existing fun/grad/hess infrastructure.
		// Note: ScalarFunction.FACTORY is a shared static singleton that retains state
		// across calls, so we explicitly construct a fresh factory to avoid leaking
		// settings between successive minimize() invocations.
		ScalarFunction objective = new ScalarFunction.ScalarFunctionFactory()
				.fun((x, args) -> fun.apply(x))
				.x0(x0)
				.args(null)
				.grad((x, args) -> grad.apply(x))
				.hess((x, args) -> hess.apply(x))
				.finiteDiffBounds(FiniteDifferenceBounds.unbounded(nVars))
				.build();

		// Counters: ScalarFunction's internal counters only see the calls we route
		// through it (just initial-value evaluations below). For accurate counts we
		// wrap every closure call.
		final int[] funCalls = {0};
		final int[] gradCalls = {0};
		final int[] hessCalls = {0};
		final Function<Matrix, Double> funCounted = x -> {
			funCalls[0]++;
			return fun.apply(x);
		};
		final Function<Matrix, Matrix> gradCounted = x -> {
			gradCalls[0]++;
			return grad.apply(x);
		};
		final Function<Matrix, Matrix> hessCounted = x -> {
			hessCalls[0]++;
			return hess.apply(x);
		};

		// Initial values from the objective and (optional) constraint.
		double f0 = objective.f();
		Matrix g0 = objective.g();
		final Matrix c0;
		final Matrix j0;
		if (eq == null || nEq == 0) {
			c0 = Matrix.Factory.zeros(0, 1);
			j0 = Matrix.Factory.zeros(0, nVars);
		} else {
			c0 = eqConstr.constrEq(x0);
			j0 = eqJac.jacEq(x0);
		}

		// Build the (fun, c_eq) and (grad, J_eq) closures the SQP expects.
		final Matrix emptyC = c0;
		final Matrix emptyJ = j0;
		IFunctionAndConstraint funAndConstr = x -> {
			double f = funCounted.apply(x);
			Matrix c = (eqConstr == null || nEq == 0) ? emptyC : eqConstr.constrEq(x);
			return new FunctionAndConstraint(f, c);
		};
		IGradientAndJacobian gradAndJac = x -> {
			Matrix g = gradCounted.apply(x);
			Matrix J = (eqJac == null || nEq == 0) ? emptyJ : eqJac.jacEq(x);
			return new GradientAndJacobian(g, J);
		};

		// Lagrangian Hessian: H_objective(x) + sum_i v[i] * H_{c_i}(x).
		// LinearConstraint contributes 0; NonlinearConstraint contributes via its
		// optional hess(x, v) callable. For pure-equality dispatch, v is the
		// equality-multiplier vector with no inequality component.
		final Object eqRef = eq;
		LagrangeHessian lagrHess = (x, v) -> {
			Matrix H = hessCounted.apply(x);
			Matrix contribution = lagrangianConstraintContribution(eqRef, x,
					colToArray(v), new double[0]);
			final Matrix Hfinal = (contribution == null) ? H : H.plus(contribution);
			return p -> Hfinal.mtimes(p);
		};

		// Initial state and trivial stopping criterion (gtol / xtol / maxIter).
		State state = new State();
		state.numConstraintEval = new int[0];
		state.numConstraintJacobianEval = new int[0];
		state.numConstraintHessianEval = new int[0];
		state.v = new Matrix[0];
		state.constr = new Matrix[0];
		state.jac = new Matrix[0];

		final long startTime = System.nanoTime();
		StoppingCriterion stop = (s, x, lastIterFailed, optimality, constrViolation,
				trustRadius, penalty, cgInfo) -> {
			s.nIter++;
			s.executionTime = System.nanoTime() - startTime;
			s.optimality = optimality;
			s.constrViolation = constrViolation;
			s.trustRadius = trustRadius;
			s.constraintPenalty = penalty;
			s.cgNIter += cgInfo.niter;
			s.cgStopCond = cgInfo.stopCond;
			s.x = x;
			if (callback != null && callback.shouldTerminate(s)) return true;
			if (optimality < gtol && constrViolation < gtol) return true;
			if (trustRadius < xtol) return true;
			if (s.nIter >= maxIter) return true;
			return false;
		};

		// Run the SQP loop.
		LinearOperator scaling = EqualityConstrainedSQP.defaultScaling(nVars);
		ProjectionMethod fact = (factorizationMethod == null)
				? ProjectionMethod.QR_FACTORIZATION : factorizationMethod;
		StatefulResult sr = EqualityConstrainedSQP.eqSQP(
				funAndConstr, gradAndJac, lagrHess,
				x0, f0, g0, c0, j0,
				stop, state,
				initialPenalty, initialTrustRadius, fact,
				null, null,                  // trustLb/Ub: unconstrained
				scaling);

		// Populate OptimizeResult -- re-evaluate the objective at the final x to
		// keep r.fun and r.grad consistent with r.x. (ScalarFunction internal
		// state is only tracked for evaluations the SQP routes through it; our
		// closures bypass that.)
		OptimizeResult r = new OptimizeResult();
		r.x = sr.x();
		r.fun = fun.apply(sr.x());
		r.grad = grad.apply(sr.x());
		// Compute the final Lagrangian gradient: grad + A^T v. With no constraint
		// (or no equality rows), v = 0, and lagrangianGrad reduces to grad.
		if (eqJac != null && nEq > 0) {
			Matrix Jx = eqJac.jacEq(sr.x());
			// Recover v from the SQP final state: at convergence,
			// optimality = ||grad + A^T v||_inf, but state doesn't expose v
			// directly. We re-derive via least-squares: v = -inv(A A^T) A grad.
			// AAt may be singular (e.g. degenerate-constraint cases like
			// scipy test_issue_18882) -- fall back to lagrangianGrad = grad.
			try {
				DenseMatrix AAt = DenseMatrix.copyFromMatrix(Jx.mtimes(Jx.transpose()));
				DenseMatrix Ag = DenseMatrix.copyFromMatrix(Jx.mtimes(r.grad));
				DenseMatrix v = LinAlg.solve(AAt, Ag).times(-1);
				r.lagrangianGrad = r.grad.plus(Jx.transpose().mtimes(v));
			} catch (RuntimeException ex) {
				r.lagrangianGrad = Matrix.Factory.copyFromMatrix(r.grad);
			}
		} else {
			r.lagrangianGrad = Matrix.Factory.copyFromMatrix(r.grad);
		}
		r.optimality = state.optimality;
		r.constraintViolation = state.constrViolation;
		r.nIter = state.nIter;
		r.numFunctionEval = funCalls[0];
		r.numJacobianEval = gradCalls[0];
		r.numHessianEval = hessCalls[0];
		r.cgIter = state.cgNIter;
		r.cgStopCond = state.cgStopCond == null ? PCGStoppingCondition.NOT_EVALUATED : state.cgStopCond;
		r.method = TrustConstrMethod.EQUALITY_CONSTRAINED_SQP;
		r.trustRadius = state.trustRadius;
		r.constraintPenalty = state.constraintPenalty;
		r.executionTime = state.executionTime;
		// Status: 1 = gtol satisfied, 2 = xtol satisfied, 0 = max iterations
		if (state.optimality < gtol && state.constrViolation < gtol) {
			r.status = 1;
			r.message = "`gtol` termination condition is satisfied.";
		} else if (state.trustRadius < xtol) {
			r.status = 2;
			r.message = "`xtol` termination condition is satisfied.";
		} else {
			r.status = 0;
			r.message = "The maximum number of function evaluations is exceeded.";
		}
		r.success = (r.status == 1)
				|| (r.status == 2 && state.constrViolation < gtol);
		return r;
	}

	/**
	 * General-purpose entry point: routes to
	 * {@link #minimizeEqualityConstrained} when the constraint has only equality
	 * rows, and to a {@code TrustRegionInteriorPoint}-driven path when any
	 * inequality rows are present. Pure unconstrained problems route through
	 * the equality path with an empty constraint set.
	 *
	 * <p>Like {@link #minimizeEqualityConstrained}, this is a focused subset of
	 * the full {@code scipy.optimize.minimize(method='trust-constr')} API: a
	 * single {@link LinearConstraint} or {@link NonlinearConstraint} (or
	 * {@code null}), no {@link Bounds}, no callback, no finite-difference
	 * Hessians/Jacobians.
	 *
	 * @param fun        objective function {@code f : R^n -> R}
	 * @param grad       analytic gradient {@code g : R^n -> R^n}
	 * @param hess       analytic Hessian {@code H : R^n -> R^{nxn}}
	 * @param x0         initial point ({@code n x 1})
	 * @param constraint single constraint or {@code null} for unconstrained
	 * @param maxIter    maximum outer iterations
	 * @param xtol       stop when {@code trustRadius < xtol}
	 * @param gtol       stop when {@code optimality < gtol} and {@code constrViolation < gtol}
	 * @param <C>        constraint type, must implement both {@link Constraint} and {@link Jacobian}
	 * @return populated {@link OptimizeResult}
	 */
	public static <C extends Constraint & Jacobian> OptimizeResult minimize(
			Function<Matrix, Double> fun,
			Function<Matrix, Matrix> grad,
			Function<Matrix, Matrix> hess,
			Matrix x0, C constraint,
			int maxIter, double xtol, double gtol) {
		validateConstraintShape(constraint, x0.getRowCount(), null);
		validateKeepFeasibleAtStart(constraint, x0);
		int nIneq = nIneqOf(constraint);
		if (nIneq == 0) {
			return minimizeEqualityConstrained(fun, grad, hess, x0, constraint, maxIter, xtol, gtol);
		}
		return minimizeInequalityConstrained(fun, grad, hess, x0, constraint, maxIter, xtol, gtol);
	}

	/**
	 * Reject equality-constraint counts greater than the variable count
	 * before the algorithm hits an array-bounds failure inside
	 * {@code Projections.qrFactorizationProjections}. Mirrors scipy's
	 * gh-20665 fix: report "more equality constraints than independent
	 * variables" with a recoverable workaround. {@code SVD_FACTORIZATION}
	 * tolerates the over-determined case (rank-revealing decomposition
	 * with implicit pseudoinverse), so the check is suppressed when SVD
	 * is the chosen factorization.
	 */
	private static void validateConstraintShape(Object constraint, long nVars,
			ProjectionMethod factorization) {
		if (factorization == ProjectionMethod.SVD_FACTORIZATION) return;
		int nEq = nEqOf(constraint);
		if (nEq > nVars) {
			throw new IllegalArgumentException(
					"Number of equality constraints (" + nEq
							+ ") is more than independent variables (" + nVars + ")."
							+ " Reformulate the problem with fewer redundant equality"
							+ " constraints, or pass factorizationMethod="
							+ "SVD_FACTORIZATION via minimizeTrustConstr(...) to use"
							+ " SVD instead.");
		}
	}

	/**
	 * Per-canonical-ineq-row {@code enforceFeasibility} flags from the
	 * supplied constraint, in the same order they appear in
	 * {@code constrIneq(x)}. Returns {@code null} for unconstrained or
	 * unrecognised types so the caller can default to all-false.
	 */
	private static boolean[] enforceFeasibilityIneqOf(Object constraint) {
		if (constraint == null) return null;
		if (constraint instanceof LinearConstraint) {
			return ((LinearConstraint) constraint).enforceFeasibilityIneq();
		}
		if (constraint instanceof NonlinearConstraint) {
			return ((NonlinearConstraint) constraint).enforceFeasibilityIneq();
		}
		if (constraint instanceof CombinedConstraint) {
			return ((CombinedConstraint) constraint).enforceFeasibilityIneq();
		}
		if (constraint instanceof SparsityForcedConstraint) {
			return ((SparsityForcedConstraint) constraint).enforceFeasibilityIneq();
		}
		return null;
	}

	/**
	 * Walk a constraint reference (single or combined) and check every row
	 * marked {@code keep_feasible=true} for strict feasibility at {@code x0}.
	 * Mirrors scipy's behaviour: keep_feasible rows must be satisfied at the
	 * start because the algorithm doesn't repair an infeasible kf row mid-run.
	 */
	private static void validateKeepFeasibleAtStart(Object constraint, Matrix x0) {
		if (constraint == null) return;
		if (constraint instanceof LinearConstraint) {
			((LinearConstraint) constraint).validateKeepFeasibleAtStart(x0);
		} else if (constraint instanceof NonlinearConstraint) {
			((NonlinearConstraint) constraint).validateKeepFeasibleAtStart(x0);
		} else if (constraint instanceof CombinedConstraint) {
			((CombinedConstraint) constraint).validateKeepFeasibleAtStart(x0);
		} else if (constraint instanceof SparsityForcedConstraint) {
			((SparsityForcedConstraint) constraint).validateKeepFeasibleAtStart(x0);
		}
	}

	/**
	 * Most general convenience overload: only the objective function is supplied.
	 * The gradient is approximated by 2-point finite differences and the Hessian
	 * by a fresh BFGS update strategy. Mirrors scipy's default behaviour when
	 * both {@code jac} and {@code hess} are omitted.
	 *
	 * @param fun        objective function {@code f : R^n -> R}
	 * @param x0         initial point ({@code n x 1})
	 * @param constraint single constraint or {@code null}
	 * @param maxIter    maximum outer iterations
	 * @param xtol       stop when {@code trustRadius < xtol}
	 * @param gtol       stop when {@code optimality < gtol} and {@code constrViolation < gtol}
	 * @param <C>        constraint type, must implement {@link Constraint} and {@link Jacobian}
	 * @return populated {@link OptimizeResult}
	 */
	public static <C extends Constraint & Jacobian> OptimizeResult minimize(
			Function<Matrix, Double> fun,
			Matrix x0, C constraint,
			int maxIter, double xtol, double gtol) {
		return minimize(fun, buildFdGrad(fun, x0, null, null), x0, constraint, maxIter, xtol, gtol);
	}

	/**
	 * Build a 2-point finite-difference gradient closure for {@code fun}.
	 * Mirrors scipy's default behaviour when {@code jac} is omitted.
	 *
	 * @param relStep optional per-component relative step size (an
	 *                {@code n x 1} column matrix) -- when non-null, passed
	 *                through to {@link NumDiff} as
	 *                {@code finiteDifferenceRelStep}. Mirrors scipy's
	 *                {@code finite_diff_rel_step}. {@code null} uses the
	 *                NumDiff default ({@code eps^(1/2)} for 2-point).
	 * @param fdBounds optional FD bounds -- when non-null, FD perturbations
	 *                stay within them. Pass {@link
	 *                StrictBounds} for
	 *                keep_feasible enforcement; {@code null} or {@code
	 *                FiniteDifferenceBounds.unbounded(...)} for no clipping.
	 *                Mirrors scipy's behaviour when {@code Bounds(keep_feasible=True)}
	 *                is supplied -- gh-11649.
	 */
	private static Function<Matrix, Matrix> buildFdGrad(
			Function<Matrix, Double> fun, Matrix x0, Matrix relStep,
			FiniteDifferenceBounds fdBounds) {
		return buildFdGrad(fun, x0, relStep, fdBounds, null);
	}

	/**
	 * Variant of {@link #buildFdGrad(java.util.function.Function, Matrix, Matrix,
	 * FiniteDifferenceBounds)} that
	 * dispatches the per-column FD evaluations through {@code workers} when
	 * non-null. Mirrors scipy's {@code workers=N} keyword on
	 * {@code _minimize_trustregion_constr}.
	 */
	private static Function<Matrix, Matrix> buildFdGrad(
			Function<Matrix, Double> fun, Matrix x0, Matrix relStep,
			FiniteDifferenceBounds fdBounds,
			ExecutorService workers) {
		// Build a fresh factory per call -- the static FACTORY singleton retains
		// state between invocations (e.g. .bounds() throws "bounds have already
		// been specified" on the second call), same hazard as ScalarFunction.FACTORY.
		final FiniteDifferenceBounds bounds =
				(fdBounds != null) ? fdBounds
						: FiniteDifferenceBounds
								.unbounded(x0.getRowCount());
		FiniteDifferenceOptions.FiniteDifferenceOptionsFactory factory =
				new FiniteDifferenceOptions.FiniteDifferenceOptionsFactory()
						.method(FiniteDifferenceMethod.TWO_POINT)
						.bounds(bounds);
		if (relStep != null) {
			factory = factory.relStep(relStep);
		}
		final FiniteDifferenceOptions options = factory.build();
		final Function<Matrix, Matrix> funVec = m ->
				DenseMatrix.column(fun.apply(m));
		final BiFunction<Matrix, Object, Matrix> funBi = (xx, args) -> funVec.apply(xx);
		if (workers == null) {
			final ToDoubleFunction<Matrix> funScalar = m -> fun.apply(m);
			return x -> {
				Matrix jac = NumDiff.approxDerivative(funScalar, x, fun.apply(x), options);
				return jac.transpose();
			};
		}
		return x -> {
			Matrix f0 = funVec.apply(x);
			Matrix jac = NumDiff.approxDerivativeWithWorkers(
					funBi, x, f0, options, null, workers);
			return jac.transpose();
		};
	}

	/**
	 * Build a Hessian closure backed by an iterative quasi-Newton strategy.
	 * The strategy is updated by the orchestrator's call to {@code grad(x)}
	 * before each Hessian evaluation: we track the previous (x, grad) pair
	 * and apply {@link org.scipy.optimize.minimize.interfaces.HessianUpdateStrategy#update}
	 * with the deltas. Returns {@code strategy.getMatrix()} for the dense
	 * Hessian Matrix expected by the orchestrator.
	 */
	private static Function<Matrix, Matrix> buildStrategyHess(
			Function<Matrix, Matrix> grad,
			Matrix x0,
			HessianUpdateStrategy strategy) {
		strategy.initialize(x0.getRowCount(),
				HessianApproximationType.HESSIAN);
		final Matrix[] xPrev = {null};
		final Matrix[] gPrev = {null};
		return x -> {
			Matrix g = grad.apply(x);
			if (xPrev[0] != null) {
				Matrix dx = x.minus(xPrev[0]);
				Matrix dg = g.minus(gPrev[0]);
				strategy.update(dx, dg);
			}
			xPrev[0] = Matrix.Factory.copyFromMatrix(x);
			gPrev[0] = Matrix.Factory.copyFromMatrix(g);
			return strategy.getMatrix();
		};
	}

	/**
	 * Convenience overload that approximates the objective Hessian via
	 * a fresh {@link BFGS} update strategy -- the scipy-default behaviour
	 * when {@code hess} is omitted from the user call.
	 *
	 * <p>The BFGS update is driven manually inside the {@code lagrHess}
	 * closure: every time the SQP / IP outer loop asks for a fresh
	 * Lagrangian Hessian (i.e. after a successful step), we compare the
	 * current x and gradient to the previous values and call
	 * {@link org.scipy.optimize.minimize.interfaces.HessianUpdateStrategy#update}
	 * to refresh the approximation.
	 *
	 * @param fun        objective function {@code f : R^n -> R}
	 * @param grad       analytic gradient {@code g : R^n -> R^n}
	 * @param x0         initial point ({@code n x 1})
	 * @param constraint single constraint or {@code null}
	 * @param maxIter    maximum outer iterations
	 * @param xtol       stop when {@code trustRadius < xtol}
	 * @param gtol       stop when {@code optimality < gtol} and {@code constrViolation < gtol}
	 * @param <C>        constraint type, must implement {@link Constraint} and {@link Jacobian}
	 * @return populated {@link OptimizeResult}
	 */
	public static <C extends Constraint & Jacobian> OptimizeResult minimize(
			Function<Matrix, Double> fun,
			Function<Matrix, Matrix> grad,
			Matrix x0, C constraint,
			int maxIter, double xtol, double gtol) {
		return minimize(fun, grad, BFGS.FACTORY.build(), x0, constraint,
				maxIter, xtol, gtol);
	}

	/**
	 * Convenience overload accepting any {@link
	 * org.scipy.optimize.minimize.interfaces.HessianUpdateStrategy} (BFGS,
	 * SR1, custom). The strategy is updated by the orchestrator after each
	 * successful iteration; users typically pass {@code BFGS.FACTORY.build()}
	 * or {@code SR1.FACTORY.build()}.
	 *
	 * @param fun        objective function {@code f : R^n -> R}
	 * @param grad       analytic gradient
	 * @param strategy   Hessian-update strategy (BFGS, SR1, ...)
	 * @param x0         initial point ({@code n x 1})
	 * @param constraint single constraint or {@code null}
	 * @param maxIter    maximum outer iterations
	 * @param xtol       stop when {@code trustRadius < xtol}
	 * @param gtol       stop when {@code optimality < gtol} and {@code constrViolation < gtol}
	 * @param <C>        constraint type, must implement {@link Constraint} and {@link Jacobian}
	 * @return populated {@link OptimizeResult}
	 */
	public static <C extends Constraint & Jacobian> OptimizeResult minimize(
			Function<Matrix, Double> fun,
			Function<Matrix, Matrix> grad,
			HessianUpdateStrategy strategy,
			Matrix x0, C constraint,
			int maxIter, double xtol, double gtol) {
		Function<Matrix, Matrix> hess = buildStrategyHess(grad, x0, strategy);
		return minimize(fun, grad, hess, x0, constraint, maxIter, xtol, gtol);
	}

	/**
	 * Multi-constraint entry point: combines the supplied constraints into a
	 * {@link CombinedConstraint} and dispatches as in
	 * {@link #minimize(java.util.function.Function, java.util.function.Function, java.util.function.Function, Matrix, Constraint, int, double, double)
	 *  the single-constraint overload}. Pass {@code null} or an empty array
	 * for unconstrained problems.
	 *
	 * @param fun         objective function {@code f : R^n -> R}
	 * @param grad        analytic gradient
	 * @param hess        analytic Hessian
	 * @param x0          initial point ({@code n x 1})
	 * @param constraints array of {@link LinearConstraint} / {@link NonlinearConstraint} (may be {@code null})
	 * @param maxIter     maximum outer iterations
	 * @param xtol        stop when {@code trustRadius < xtol}
	 * @param gtol        stop when {@code optimality < gtol} and {@code constrViolation < gtol}
	 * @return populated {@link OptimizeResult}
	 */
	public static OptimizeResult minimize(
			Function<Matrix, Double> fun,
			Function<Matrix, Matrix> grad,
			Function<Matrix, Matrix> hess,
			Matrix x0, Object[] constraints,
			int maxIter, double xtol, double gtol) {
		int nVars = (int) x0.getRowCount();
		if (constraints == null || constraints.length == 0) {
			return minimizeEqualityConstrained(fun, grad, hess, x0, null, maxIter, xtol, gtol);
		}
		for (Object c : constraints) {
			validateConstraintShape(c, x0.getRowCount(), null);
			validateKeepFeasibleAtStart(c, x0);
		}
		// Also validate the combined eq-row count: even if no single source
		// has nEq > nVars, the concatenation might.
		if (constraints.length > 1) {
			int totalEq = 0;
			for (Object c : constraints) totalEq += nEqOf(c);
			if (totalEq > nVars) {
				throw new IllegalArgumentException(
						"Number of equality constraints (" + totalEq
								+ ") is more than independent variables (" + nVars + ")."
								+ " Reformulate the problem with fewer redundant equality"
								+ " constraints, or pass factorizationMethod="
								+ "SVD_FACTORIZATION via minimizeTrustConstr(...) to use"
								+ " SVD instead.");
			}
		}
		if (constraints.length == 1) {
			Object c = constraints[0];
			if (c instanceof LinearConstraint) {
				return minimize(fun, grad, hess, x0, (LinearConstraint) c, maxIter, xtol, gtol);
			}
			if (c instanceof NonlinearConstraint) {
				return minimize(fun, grad, hess, x0, (NonlinearConstraint) c, maxIter, xtol, gtol);
			}
		}
		CombinedConstraint combined = new CombinedConstraint(constraints, nVars);
		if (combined.nIneq() == 0) {
			return minimizeEqualityConstrainedRaw(fun, grad, hess, x0, combined,
					combined.nEq(), maxIter, xtol, gtol, null, 1.0, 1.0, null);
		}
		return minimizeInequalityConstrainedRaw(fun, grad, hess, x0, combined,
				combined.nEq(), combined.nIneq(), maxIter, xtol, gtol, gtol, null,
				1.0, 1.0, 0.1, 0.1, null);
	}

	private static int nIneqOf(Object constraint) {
		if (constraint == null) return 0;
		if (constraint instanceof LinearConstraint) return ((LinearConstraint) constraint).nIneq();
		if (constraint instanceof NonlinearConstraint) return ((NonlinearConstraint) constraint).nIneq();
		if (constraint instanceof CombinedConstraint) return ((CombinedConstraint) constraint).nIneq();
		if (constraint instanceof SparsityForcedConstraint) return ((SparsityForcedConstraint) constraint).nIneq();
		throw new IllegalArgumentException("Unsupported constraint type: " + constraint.getClass().getName());
	}

	/**
	 * Sum of constraint Hessian-of-Lagrangian contributions across the (possibly
	 * combined) constraint. {@link LinearConstraint} contributes 0;
	 * {@link NonlinearConstraint} contributes via its
	 * {@link NonlinearConstraint#lagrangianContribution} callable when set.
	 * Returns {@code null} if no constraint contributes a Hessian -- the caller
	 * then uses the objective Hessian alone.
	 */
	private static Matrix lagrangianConstraintContribution(Object constraint, Matrix x,
			double[] vEq, double[] vIneq) {
		if (constraint == null) return null;
		if (constraint instanceof LinearConstraint) return null;
		if (constraint instanceof NonlinearConstraint) {
			return ((NonlinearConstraint) constraint).lagrangianContribution(x, vEq, vIneq);
		}
		if (constraint instanceof CombinedConstraint) {
			return ((CombinedConstraint) constraint).lagrangianContribution(x, vEq, vIneq);
		}
		if (constraint instanceof SparsityForcedConstraint) {
			return ((SparsityForcedConstraint) constraint).lagrangianContribution(x, vEq, vIneq);
		}
		return null;
	}

	/** Extract a column vector ({@code n x 1}) as a {@code double[n]}. */
	private static double[] colToArray(Matrix v) {
		long rows = v.getRowCount();
		double[] out = new double[(int) rows];
		for (int i = 0; i < rows; ++i) {
			out[i] = v.getAsDouble(i, 0);
		}
		return out;
	}

	private static int nEqOf(Object constraint) {
		if (constraint == null) return 0;
		if (constraint instanceof LinearConstraint) return ((LinearConstraint) constraint).nEq();
		if (constraint instanceof NonlinearConstraint) return ((NonlinearConstraint) constraint).nEq();
		if (constraint instanceof CombinedConstraint) return ((CombinedConstraint) constraint).nEq();
		if (constraint instanceof SparsityForcedConstraint) return ((SparsityForcedConstraint) constraint).nEq();
		throw new IllegalArgumentException("Unsupported constraint type: " + constraint.getClass().getName());
	}

	/**
	 * Inequality-constrained dispatch: invokes
	 * {@link TrustRegionInteriorPoint#trustRegionInteriorPoint} with the
	 * constraint's eq + ineq rows, mirroring the {@code 'tr_interior_point'}
	 * branch of scipy's {@code _minimize_trustregion_constr}. Equality rows
	 * are passed through alongside the inequalities; this means a constraint
	 * with both kinds of rows still works.
	 */
	private static <C extends Constraint & Jacobian> OptimizeResult minimizeInequalityConstrained(
			Function<Matrix, Double> fun,
			Function<Matrix, Matrix> grad,
			Function<Matrix, Matrix> hess,
			Matrix x0, C constraint,
			int maxIter, double xtol, double gtol) {
		return minimizeInequalityConstrainedRaw(fun, grad, hess, x0, constraint,
				nEqOf(constraint), nIneqOf(constraint), maxIter, xtol, gtol, gtol, null,
				1.0, 1.0, 0.1, 0.1, null);
	}

	private static OptimizeResult minimizeInequalityConstrainedRaw(
			Function<Matrix, Double> fun,
			Function<Matrix, Matrix> grad,
			Function<Matrix, Matrix> hess,
			Matrix x0, Object constraint, int nEq, int nIneq,
			int maxIter, double xtol, double gtol, double barrierTol,
			IterationCallback callback,
			double initialPenalty,
			double initialTrustRadius,
			double initialBarrierParameter,
			double initialBarrierTolerance,
			ProjectionMethod factorizationMethod) {
		final int nVars = (int) x0.getRowCount();
		final Constraint constr = (Constraint) constraint;
		final Jacobian jac = (Jacobian) constraint;

		// Counters for accurate eval reporting (scipy convention).
		final int[] funCalls = {0};
		final int[] gradCalls = {0};
		final int[] hessCalls = {0};

		// Wrap fun/grad with a no-op args parameter and increment counters on each call.
		final ToDoubleBiFunction<Matrix, Object> funBi = (x, args) -> {
			funCalls[0]++;
			return fun.apply(x);
		};
		final BiFunction<Matrix, Object, Matrix> gradBi = (x, args) -> {
			gradCalls[0]++;
			return grad.apply(x);
		};

		// Lagrangian Hessian: H_objective(x) + sum_i v[i] * H_{c_i}(x). The IP path
		// passes v with eq multipliers first and ineq second, so we slice
		// accordingly before delegating to the constraint's Hessian-of-Lagrangian
		// callable (NonlinearConstraint.lagrangianContribution); LinearConstraint
		// contributions are 0 by construction.
		final Object constraintRef = constraint;
		final int nEqLocal = nEq;
		final int nIneqLocal = nIneq;
		LagrangeHessian lagrHess = (x, v) -> {
			hessCalls[0]++;
			Matrix H = hess.apply(x);
			double[] vAll = colToArray(v);
			double[] vEq = new double[nEqLocal];
			double[] vIneq = new double[nIneqLocal];
			System.arraycopy(vAll, 0, vEq, 0, Math.min(nEqLocal, vAll.length));
			if (vAll.length >= nEqLocal + nIneqLocal) {
				System.arraycopy(vAll, nEqLocal, vIneq, 0, nIneqLocal);
			}
			Matrix contribution = lagrangianConstraintContribution(constraintRef, x, vEq, vIneq);
			final Matrix Hfinal = (contribution == null) ? H : H.plus(contribution);
			return p -> Hfinal.mtimes(p);
		};

		// Initial values
		funCalls[0]++;
		double f0 = fun.apply(x0);
		gradCalls[0]++;
		Matrix g0 = grad.apply(x0);
		Matrix cIneq0 = constr.constrIneq(x0);
		Matrix jIneq0 = jac.jacIneq(x0);
		Matrix cEq0 = nEq > 0 ? constr.constrEq(x0) : Matrix.Factory.zeros(0, 1);
		Matrix jEq0 = nEq > 0 ? jac.jacEq(x0) : Matrix.Factory.zeros(0, nVars);

		// Initial state
		State state = new State();
		state.numConstraintEval = new int[1];
		state.numConstraintJacobianEval = new int[1];
		state.numConstraintHessianEval = new int[1];
		state.v = new Matrix[1];
		state.constr = new Matrix[1];
		state.jac = new Matrix[1];

		final long startTime = System.nanoTime();
		// Per-iteration trackers: the GlobalStoppingCriteria signature passes
		// barrierParameter/barrierTolerance per call but State doesn't store
		// them (only StateIP does). Capture the latest values so the result
		// can surface them.
		final double[] lastBarrierParameter = {Double.NaN};
		final double[] lastBarrierTolerance = {Double.NaN};
		GlobalStoppingCriteria stop = (s, x, lastIterFailed, optimality, constrViolation,
				trustRadius, penalty, cgInfo, barrierParameter, barrierTolerance) -> {
			s.nIter++;
			s.executionTime = System.nanoTime() - startTime;
			s.optimality = optimality;
			s.constrViolation = constrViolation;
			s.trustRadius = trustRadius;
			s.constraintPenalty = penalty;
			s.cgNIter += cgInfo.niter;
			s.cgStopCond = cgInfo.stopCond;
			s.x = x;
			lastBarrierParameter[0] = barrierParameter;
			lastBarrierTolerance[0] = barrierTolerance;
			if (callback != null && callback.shouldTerminate(s)) return true;
			if (optimality < gtol && constrViolation < gtol) return true;
			if (trustRadius < xtol && barrierParameter < barrierTol) return true;
			if (s.nIter >= maxIter) return true;
			return false;
		};

		ProjectionMethod ipFact = (factorizationMethod == null)
				? ProjectionMethod.AUGMENTED_SYSTEM : factorizationMethod;
		// Extract per-canonical-ineq-row enforceFeasibility flags from the
		// constraint(s). The IP path's BarrierSubproblem.computeFunction uses
		// these to drive the slack to make `c_i(x) - 0` (i.e. the canonical
		// ineq value) exactly zero -- pushing the algorithm away from
		// infeasible interior steps for those rows. Mirrors scipy's
		// keep_feasible enforcement on the IP path.
		boolean[] enforceFeasibility = enforceFeasibilityIneqOf(constraint);
		if (enforceFeasibility == null || enforceFeasibility.length != nIneq) {
			enforceFeasibility = new boolean[nIneq];
		}
		StatefulResult sr = TrustRegionInteriorPoint.trustRegionInteriorPoint(
				funBi, gradBi, lagrHess,
				nVars, nIneq, nEq,
				constr, jac,
				x0, f0, g0,
				cIneq0, jIneq0, cEq0, jEq0,
				stop,
				enforceFeasibility,
				xtol, state,
				initialBarrierParameter, initialBarrierTolerance,
				initialPenalty, initialTrustRadius,
				ipFact);

		// Populate result
		OptimizeResult r = new OptimizeResult();
		r.x = sr.x();
		r.fun = fun.apply(sr.x());
		r.grad = grad.apply(sr.x());
		// Final Lagrangian gradient at sr.x():
		//     g + Jeq^T v_eq + Jineq^T lambda_ineq.
		// TrustRegionInteriorPoint exposes the augmented-system multiplier from
		// the last barrier subproblem via sr.v(): length nEq + nIneq, with
		// equality multipliers in [0, nEq) and slack-row multipliers -- which
		// equal the original problem's lambda -- in [nEq, nEq+nIneq).
		Matrix lagrGrad = Matrix.Factory.copyFromMatrix(r.grad);
		Matrix vAll = sr.v();
		if (vAll != null && nEq > 0) {
			Matrix vEq = vAll.subMatrix(0, 0, nEq - 1, 0);
			Matrix Jeq = jac.jacEq(sr.x());
			lagrGrad = lagrGrad.plus(Jeq.transpose().mtimes(vEq));
		}
		if (vAll != null && nIneq > 0) {
			Matrix vIneq = vAll.subMatrix(nEq, 0, nEq + nIneq - 1, 0);
			Matrix Jineq = jac.jacIneq(sr.x());
			lagrGrad = lagrGrad.plus(Jineq.transpose().mtimes(vIneq));
		}
		r.lagrangianGrad = lagrGrad;
		r.numFunctionEval = funCalls[0];
		r.numJacobianEval = gradCalls[0];
		r.numHessianEval = hessCalls[0];
		r.optimality = state.optimality;
		r.constraintViolation = state.constrViolation;
		r.nIter = state.nIter;
		r.cgIter = state.cgNIter;
		r.cgStopCond = state.cgStopCond == null ? PCGStoppingCondition.NOT_EVALUATED : state.cgStopCond;
		r.method = TrustConstrMethod.TRUST_REGION_INTERIOR_POINT;
		r.trustRadius = state.trustRadius;
		r.constraintPenalty = state.constraintPenalty;
		r.barrierParameter = lastBarrierParameter[0];
		r.barrierTolerance = lastBarrierTolerance[0];
		r.executionTime = state.executionTime;
		if (state.optimality < gtol && state.constrViolation < gtol) {
			r.status = 1;
			r.message = "`gtol` termination condition is satisfied.";
		} else if (state.trustRadius < xtol) {
			r.status = 2;
			r.message = "`xtol` termination condition is satisfied.";
		} else {
			r.status = 0;
			r.message = "The maximum number of function evaluations is exceeded.";
		}
		r.success = (r.status == 1)
				|| (r.status == 2 && state.constrViolation < gtol);
		return r;
	}

	/**
	 * Minimize a scalar function subject to constraints. Full scipy-shape
	 * entry point: see Conn, Gould &amp; Toint, <i>Trust Region Methods</i>
	 * (SIAM, 2000), p.19 for the algorithm and parameter recommendations.
	 *
	 * @param fun       objective {@code (x, args) -> f(x)}
	 * @param x0        starting point ({@code n x 1})
	 * @param args      extra arguments forwarded to {@code fun}/{@code grad}/{@code hess}; may be {@code null}
	 * @param grad      gradient {@code (x, args) -> gradf(x)}; if {@code null}, computed by 2-point finite differences
	 * @param hess      Hessian {@code (x, args) -> grad^2f(x)}; if {@code null}, approximated by BFGS
	 * @param hessp     Hessian-vector product; honored when {@code hess} is {@code null} (materialised via
	 *                  {@link HessianLinearOperator})
	 * @param bounds    box bounds on {@code x}; folded into the constraint set as a {@link LinearConstraint};
	 *                  may be {@code null}
	 * @param constraints either a single {@link LinearConstraint} / {@link NonlinearConstraint}, an
	 *                  {@code Object[]} of such, or {@code null}
	 * @param xTol      tolerance for termination by change in {@code x}: stop when {@code tr_radius < xTol}
	 * @param gTol      tolerance for termination by Lagrangian gradient norm and constraint violation
	 * @param barrierTol IP-only: termination requires barrier parameter below this threshold
	 * @param sparseJacobian {@code true}/{@code false} forces all constraint Jacobians to that
	 *                  representation via {@link SparsityForcedConstraint}; {@code Optional.empty()}
	 *                  uses the constraint's native auto-detect
	 * @param callback  per-iteration callback; return {@code true} to terminate. May be {@code null}
	 * @param maxIter   maximum outer iterations
	 * @param verbose   verbosity level (0 silent, 1 termination report, 2-3 per-iteration progress)
	 * @param finiteDifferenceRelStep relative step size for FD gradient/Hessian; may be {@code null}
	 * @param initialConstraintPenalty initial constraint penalty for the merit function {@code f(x) + rho ||c(x)||_2}
	 * @param initialTrustRadius initial trust-region radius
	 * @param initialBarrierParameter IP-only: initial barrier parameter for the log-barrier subproblem
	 * @param initialBarrierTolerance IP-only: initial inner-loop tolerance for the barrier subproblem
	 * @param factorizationMethod how to factor the equality-Jacobian for the projection: one of
	 *                  {@code AUGMENTED_SYSTEM}, {@code QR_FACTORIZATION}, {@code SVD_FACTORIZATION},
	 *                  or {@code null} for auto-select (QR for dense, AUGMENTED_SYSTEM for sparse)
	 * @param disp      if {@code true}, bumps {@code verbose} to 1 when it was 0
	 * @return populated {@link OptimizeResult}
	 */
	public static OptimizeResult minimizeTrustConstr(ToDoubleBiFunction<Matrix, Object> fun, Matrix x0, Object args,
			BiFunction<Matrix, Object, Matrix> grad, BiFunction<Matrix, Object, Matrix> hess,
			HessianProduct hessp, Bounds bounds,
			Object constraints,
			double xTol, double gTol, double barrierTol, Optional<Boolean> sparseJacobian,
			IterationCallback callback, int maxIter, int verbose, Matrix finiteDifferenceRelStep, double initialConstraintPenalty,
			double initialTrustRadius, double initialBarrierParameter, double initialBarrierTolerance,
			ProjectionMethod factorizationMethod, boolean disp) {
		return minimizeTrustConstr(fun, x0, args, grad, hess, hessp, bounds, constraints,
				xTol, gTol, barrierTol, sparseJacobian, callback, maxIter, verbose,
				finiteDifferenceRelStep, initialConstraintPenalty, initialTrustRadius,
				initialBarrierParameter, initialBarrierTolerance, factorizationMethod, disp,
				/*workers=*/ null);
	}

	/**
	 * Strict scipy {@code _minimize_trustregion_constr(..., workers=N)}
	 * counterpart. Same arguments as
	 * {@link #minimizeTrustConstr(ToDoubleBiFunction, Matrix, Object, BiFunction,
	 * BiFunction, HessianProduct, Bounds, Object, double, double, double, Optional,
	 * IterationCallback, int, int, Matrix, double, double, double, double,
	 * ProjectionMethod, boolean)}, plus a final {@link
	 * ExecutorService} for parallel FD evaluation.
	 *
	 * <p>{@code workers} only affects the FD-gradient path (when {@code grad} is
	 * {@code null}); analytic gradients are unaffected.
	 *
	 * @param fun                       objective {@code (x, args) -> f}
	 * @param x0                        starting iterate ({@code n x 1})
	 * @param args                      extra arguments forwarded to {@code fun}/{@code grad}/{@code hess}
	 * @param grad                      analytic gradient {@code (x, args) -> grad f(x)}, or {@code null} for FD
	 * @param hess                      analytic Hessian {@code (x, args) -> H(x)}, or {@code null} for BFGS
	 * @param hessp                     matrix-free Hessian-vector product {@code (x, p, args) -> H(x) p}, or {@code null}
	 * @param bounds                    box bounds on the decision variables, or {@code null}
	 * @param constraints               {@link LinearConstraint}, {@link NonlinearConstraint}, {@code Object[]} mix, or {@code null}
	 * @param xTol                      step-size convergence tolerance
	 * @param gTol                      KKT-optimality convergence tolerance
	 * @param barrierTol                IP-path barrier-parameter convergence tolerance
	 * @param sparseJacobian            forces sparse / dense Jacobian; {@code Optional.empty()} auto-detects
	 * @param callback                  per-iteration callback, returns {@code true} to terminate; {@code null} disables
	 * @param maxIter                   iteration limit
	 * @param verbose                   verbosity level (0 silent, 1 progress table)
	 * @param finiteDifferenceRelStep   FD relative step (per-element vector); {@code null} uses the eps-derived default
	 * @param initialConstraintPenalty  initial merit-function penalty
	 * @param initialTrustRadius        initial trust-region radius
	 * @param initialBarrierParameter   IP-path: initial barrier coefficient
	 * @param initialBarrierTolerance   IP-path: initial barrier-subproblem tolerance
	 * @param factorizationMethod       projection factorisation; {@code null} picks
	 *                                  QR for dense / AugmentedSystem for sparse
	 * @param disp                      bumps {@code verbose} to 1 when it was 0
	 * @param workers                   executor for parallel FD evaluation; {@code null} = serial
	 * @return populated {@link OptimizeResult}
	 */
	public static OptimizeResult minimizeTrustConstr(ToDoubleBiFunction<Matrix, Object> fun, Matrix x0, Object args,
			BiFunction<Matrix, Object, Matrix> grad, BiFunction<Matrix, Object, Matrix> hess,
			HessianProduct hessp, Bounds bounds,
			Object constraints,
			double xTol, double gTol, double barrierTol, Optional<Boolean> sparseJacobian,
			IterationCallback callback, int maxIter, int verbose, Matrix finiteDifferenceRelStep, double initialConstraintPenalty,
			double initialTrustRadius, double initialBarrierParameter, double initialBarrierTolerance,
			ProjectionMethod factorizationMethod, boolean disp,
			ExecutorService workers) {
		// Plumb workers through a thread-local read inside buildFdGrad. Set it
		// here, clear it on exit so we don't leak across calls.
		WORKERS_TL.set(workers);
		try {
			return minimizeTrustConstrImpl(fun, x0, args, grad, hess, hessp, bounds, constraints,
					xTol, gTol, barrierTol, sparseJacobian, callback, maxIter, verbose,
					finiteDifferenceRelStep, initialConstraintPenalty, initialTrustRadius,
					initialBarrierParameter, initialBarrierTolerance, factorizationMethod, disp);
		} finally {
			WORKERS_TL.remove();
		}
	}

	/** Thread-local for plumbing {@code workers} through to {@link #buildFdGrad}. */
	private static final ThreadLocal<ExecutorService> WORKERS_TL =
			new ThreadLocal<>();

	private static OptimizeResult minimizeTrustConstrImpl(ToDoubleBiFunction<Matrix, Object> fun, Matrix x0, Object args,
			BiFunction<Matrix, Object, Matrix> grad, BiFunction<Matrix, Object, Matrix> hess,
			HessianProduct hessp, Bounds bounds,
			Object constraints,
			double xTol, double gTol, double barrierTol, Optional<Boolean> sparseJacobian,
			IterationCallback callback, int maxIter, int verbose, Matrix finiteDifferenceRelStep, double initialConstraintPenalty,
			double initialTrustRadius, double initialBarrierParameter, double initialBarrierTolerance,
			ProjectionMethod factorizationMethod, boolean disp) {

		// Adapter implementation: translate the scipy-shape signature to the
		// convenience overloads (which carry the actual SQP / IP machinery)
		// by stripping the args parameter, folding bounds + constraints into a
		// single combined-constraint, and dispatching by which derivatives the
		// caller supplied.
		//
		// `sparseJacobian` is honored when explicitly set: each constraint is
		// wrapped in {@link SparsityForcedConstraint} so its `jacEq`/`jacIneq`
		// outputs are converted to the requested representation. When
		// `Optional.empty()`, the auto-detect logic in `LinearConstraint` and
		// `CombinedConstraint` decides per call.
		//
		// `hessp` is honored when hess is null (materialised via
		// HessianLinearOperator).
		final long nVars = x0.getRowCount();

		// disp=true bumps verbose to 1 if it was 0 (mirrors scipy).
		// When verbose>=1 and callback is null, install a printing callback;
		// when callback is non-null, the user's callback runs (user-supplied
		// callback takes precedence over the auto-printer).
		final int effectiveVerbose = (disp && verbose == 0) ? 1 : verbose;
		// effectiveCallback is set after constraintArr is built so the auto-
		// installed verbose printer can dispatch to SQPReport (eq-only) or
		// IPReport (any inequality). See the assignment below the totalEq
		// validation block.

		final Function<Matrix, Double> funF = x -> fun.applyAsDouble(x, args);
		final Function<Matrix, Matrix> gradF =
				(grad == null) ? null : x -> grad.apply(x, args);

		final BiFunction<Matrix, Object, Matrix> hessBi;
		if (hess != null) {
			hessBi = hess;
		} else if (hessp != null) {
			hessBi = new HessianLinearOperator(hessp, nVars);
		} else {
			hessBi = null;
		}
		final Function<Matrix, Matrix> hessF =
				(hessBi == null) ? null : x -> hessBi.apply(x, args);

		// Normalise constraints into a single Object[] (bounds prepended as a
		// LinearConstraint when supplied). null and empty are unconstrained.
		List<Object> sources = new LinkedList<>();
		if (bounds != null) {
			sources.add(LinearConstraint.fromBounds(bounds));
		}
		if (constraints != null) {
			if (constraints instanceof Object[]) {
				for (Object c : (Object[]) constraints) {
					if (c != null) sources.add(c);
				}
			} else {
				sources.add(constraints);
			}
		}
		// Honor explicit sparseJacobian: wrap each source so its jacEq/jacIneq
		// is forced to the requested representation. When Optional.empty(),
		// auto-detect logic in LinearConstraint / CombinedConstraint decides.
		if (sparseJacobian != null && sparseJacobian.isPresent()) {
			boolean wantSparse = sparseJacobian.get();
			List<Object> wrapped = new LinkedList<>();
			for (Object src : sources) {
				wrapped.add(new SparsityForcedConstraint(src, wantSparse));
			}
			sources = wrapped;
		}
		final Object[] constraintArr = sources.toArray();

		// Validate: more equality constraints than variables only works with
		// SVD_FACTORIZATION (rank-revealing). Per scipy gh-20665.
		int totalEq = 0;
		int totalIneq = 0;
		for (Object c : constraintArr) {
			validateConstraintShape(c, nVars, factorizationMethod);
			validateKeepFeasibleAtStart(c, x0);
			totalEq += nEqOf(c);
			totalIneq += nIneqOf(c);
		}

		// Auto-install verbose printer (deferred from setup-time so we can
		// dispatch to SQPReport vs IPReport). User-supplied callback always
		// takes precedence over the auto-printer.
		final IterationCallback effectiveCallback;
		if (callback != null) {
			effectiveCallback = callback;
		} else if (effectiveVerbose >= 1) {
			effectiveCallback = (totalIneq == 0)
					? makeSQPProgressCallback()
					: makeIPProgressCallback();
		} else {
			effectiveCallback = null;
		}
		if (factorizationMethod != ProjectionMethod.SVD_FACTORIZATION
				&& totalEq > nVars) {
			throw new IllegalArgumentException(
					"Number of equality constraints (" + totalEq
							+ ") is more than independent variables (" + nVars + ")."
							+ " Reformulate the problem with fewer redundant equality"
							+ " constraints, or pass factorizationMethod="
							+ "SVD_FACTORIZATION via minimizeTrustConstr(...) to use"
							+ " SVD instead.");
		}

		// Route through the raw paths whenever the caller has tweaked any
		// parameter that the convenience overloads don't expose: callback,
		// the four initial* tuning knobs, or factorizationMethod. The
		// convenience overloads remain reachable for the common case where
		// users want defaults -- but the full-shape entry point honors every
		// knob it accepts. Synthesize FD-grad / BFGS-Hess closures inline
		// if the caller omitted them.
		final boolean tuned = (effectiveCallback != null)
				|| (factorizationMethod != null)
				|| (finiteDifferenceRelStep != null)
				|| (barrierTol != gTol)
				|| (sparseJacobian != null && sparseJacobian.isPresent())
				|| (initialConstraintPenalty != 1.0)
				|| (initialTrustRadius != 1.0)
				|| (initialBarrierParameter != 0.1)
				|| (initialBarrierTolerance != 0.1);
		if (tuned) {
			// FD bounds: when the caller supplied Bounds(keep_feasible=True),
			// FD perturbations must stay inside them -- scipy gh-11649. We
			// always pass a StrictBounds wrapper when bounds are present,
			// regardless of keep_feasible, since clipping is harmless when
			// the start is feasible (the dominant case).
			final FiniteDifferenceBounds fdBounds =
					(bounds != null)
							? new StrictBounds(bounds)
							: null;
			final Function<Matrix, Matrix> gradFinal =
					(gradF != null) ? gradF
							: buildFdGrad(funF, x0, finiteDifferenceRelStep, fdBounds, WORKERS_TL.get());
			final Function<Matrix, Matrix> hessFinal;
			if (hessF != null) {
				hessFinal = hessF;
			} else {
				hessFinal = buildStrategyHess(gradFinal, x0, BFGS.FACTORY.build());
			}
			Object combined;
			if (constraintArr.length == 0) {
				combined = null;
			} else if (constraintArr.length == 1) {
				combined = constraintArr[0];
			} else {
				combined = new CombinedConstraint(constraintArr, (int) nVars);
			}
			int nIneq = nIneqOf(combined);
			int nEq = nEqOf(combined);
			if (nIneq == 0) {
				return minimizeEqualityConstrainedRaw(funF, gradFinal, hessFinal,
						x0, combined, nEq, maxIter, xTol, gTol, effectiveCallback,
						initialConstraintPenalty, initialTrustRadius, factorizationMethod);
			}
			return minimizeInequalityConstrainedRaw(funF, gradFinal, hessFinal,
					x0, combined, nEq, nIneq, maxIter, xTol, gTol, barrierTol,
					effectiveCallback,
					initialConstraintPenalty, initialTrustRadius,
					initialBarrierParameter, initialBarrierTolerance,
					factorizationMethod);
		}

		// Dispatch by which derivatives are supplied.
		if (gradF == null) {
			// Most hands-off entry: only the objective. Goes through the
			// FD-grad + BFGS-Hess overload, which itself routes by constraint
			// shape. Doesn't yet support multi-constraint or hess via this
			// path, so unwrap to the scalar form when possible.
			if (constraintArr.length == 0) {
				return minimize(funF, x0, (LinearConstraint) null, maxIter, xTol, gTol);
			}
			if (constraintArr.length == 1) {
				Object c = constraintArr[0];
				if (c instanceof LinearConstraint) {
					return minimize(funF, x0, (LinearConstraint) c, maxIter, xTol, gTol);
				}
				if (c instanceof NonlinearConstraint) {
					return minimize(funF, x0, (NonlinearConstraint) c, maxIter, xTol, gTol);
				}
			}
			// Multi-constraint: combine and route through the typed overload.
			CombinedConstraint cc = new CombinedConstraint(constraintArr, (int) nVars);
			return minimize(funF, x0, cc, maxIter, xTol, gTol);
		}

		if (hessF == null) {
			// Gradient supplied, no Hessian: BFGS for the objective.
			if (constraintArr.length == 0) {
				return minimize(funF, gradF, x0, (LinearConstraint) null, maxIter, xTol, gTol);
			}
			if (constraintArr.length == 1) {
				Object c = constraintArr[0];
				if (c instanceof LinearConstraint) {
					return minimize(funF, gradF, x0, (LinearConstraint) c, maxIter, xTol, gTol);
				}
				if (c instanceof NonlinearConstraint) {
					return minimize(funF, gradF, x0, (NonlinearConstraint) c, maxIter, xTol, gTol);
				}
			}
			CombinedConstraint cc = new CombinedConstraint(constraintArr, (int) nVars);
			return minimize(funF, gradF, x0, cc, maxIter, xTol, gTol);
		}

		// Full analytic: gradient and Hessian supplied. Single-constraint
		// path uses the typed overload; multi-constraint goes through the
		// Object[] overload which handles concatenation internally.
		if (constraintArr.length == 0) {
			return minimize(funF, gradF, hessF, x0, (LinearConstraint) null, maxIter, xTol, gTol);
		}
		if (constraintArr.length == 1) {
			Object c = constraintArr[0];
			if (c instanceof LinearConstraint) {
				return minimize(funF, gradF, hessF, x0, (LinearConstraint) c, maxIter, xTol, gTol);
			}
			if (c instanceof NonlinearConstraint) {
				return minimize(funF, gradF, hessF, x0, (NonlinearConstraint) c, maxIter, xTol, gTol);
			}
		}
		return minimize(funF, gradF, hessF, x0, constraintArr, maxIter, xTol, gTol);
	}

	/**
	 * Auto-installed verbose printer for the equality-constrained-SQP path.
	 * Renders each iteration as one row of {@link
	 * SQPReport}.
	 *
	 * @return callback that prints the SQP progress table; never returns
	 *         {@code true} (so it never requests termination)
	 */
	private static IterationCallback makeSQPProgressCallback() {
		final SQPReport report =
				new SQPReport();
		final boolean[] headerPrinted = {false};
		return state -> {
			if (!headerPrinted[0]) {
				report.printHeader();
				headerPrinted[0] = true;
			}
			int cgStop = (state.cgStopCond != null) ? state.cgStopCond.ordinal() : 0;
			report.printIteration(
					state.nIter, state.numEval, state.cgNIter,
					state.fun, state.trustRadius, state.optimality,
					state.constrViolation, state.constraintPenalty, cgStop);
			return false;
		};
	}

	/**
	 * Auto-installed verbose printer for the trust-region interior-point path.
	 * Renders each iteration as one row of {@link
	 * IPReport}.
	 *
	 * @return callback that prints the IP progress table; never returns
	 *         {@code true} (so it never requests termination)
	 */
	private static IterationCallback makeIPProgressCallback() {
		final IPReport report =
				new IPReport();
		final boolean[] headerPrinted = {false};
		return state -> {
			if (!headerPrinted[0]) {
				report.printHeader();
				headerPrinted[0] = true;
			}
			double barrier = (state instanceof StateIP ip)
					? ip.barrierParameter
					: 0.0;
			int cgStop = (state.cgStopCond != null) ? state.cgStopCond.ordinal() : 0;
			report.printIteration(
					state.nIter, state.numEval, state.cgNIter,
					state.fun, state.trustRadius, state.optimality,
					state.constrViolation, state.constraintPenalty,
					barrier, cgStop);
			return false;
		};
	}

}
