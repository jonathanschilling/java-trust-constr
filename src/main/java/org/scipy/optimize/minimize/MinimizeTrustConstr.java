package org.scipy.optimize.minimize;

import java.util.LinkedList;
import java.util.List;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.ToDoubleBiFunction;

import org.scipy.optimize.minimize.enums.PCGStoppingCondition;
import org.scipy.optimize.minimize.enums.ProjectionMethod;
import org.scipy.optimize.minimize.enums.TrustConstrMethod;
import org.scipy.optimize.minimize.interfaces.Constraint;
import org.scipy.optimize.minimize.interfaces.HessianProduct;
import org.scipy.optimize.minimize.interfaces.GlobalStoppingCriteria;
import org.scipy.optimize.minimize.interfaces.IFunctionAndConstraint;
import org.scipy.optimize.minimize.interfaces.IGradientAndJacobian;
import org.scipy.optimize.minimize.interfaces.Jacobian;
import org.scipy.optimize.minimize.interfaces.LagrangeHessian;
import org.scipy.optimize.minimize.interfaces.LinearOperator;
import org.scipy.optimize.minimize.interfaces.StoppingCriterion;
import org.scipy.optimize.minimize.records.Bounds;
import org.scipy.optimize.minimize.records.CGInfo;
import org.scipy.optimize.minimize.records.FiniteDifferenceBounds;
import org.scipy.optimize.minimize.records.FunctionAndConstraint;
import org.scipy.optimize.minimize.records.GradientAndJacobian;
import org.scipy.optimize.minimize.records.OptimizeResult;
import org.scipy.optimize.minimize.records.PreparedConstraint;
import org.scipy.optimize.minimize.records.State;
import org.scipy.optimize.minimize.records.StateIP;
import org.scipy.optimize.minimize.records.StatefulResult;
import org.scipy.optimize.minimize.records.StrictBounds;
import org.ujmp.core.Matrix;

/** Java port of scipy.optimize.minimize(method='trust-constr') */
public class MinimizeTrustConstr {

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
			VectorFunction c = preparedConstraints[i].fun();
			state.numConstraintEval[i] = c.numFunctionEvals();
			state.numConstraintJacobianEval[i] = c.numJacobianEvals();
			state.numConstraintHessianEval[i] = c.numHessianEvals();
		}

		if (!lastIterationFailed) {
			state.x = x;
			state.fun = objective.f();
			state.grad = objective.g();

			for (int i = 0; i < preparedConstraints.length; ++i) {
				VectorFunction c = preparedConstraints[i].fun();
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
	 * @return populated {@link OptimizeResult}
	 */
	public static <C extends Constraint & Jacobian> OptimizeResult minimizeEqualityConstrained(
			java.util.function.Function<Matrix, Double> fun,
			java.util.function.Function<Matrix, Matrix> grad,
			java.util.function.Function<Matrix, Matrix> hess,
			Matrix x0, C eq,
			int maxIter, double xtol, double gtol) {
		if (nIneqOf(eq) != 0) {
			throw new UnsupportedOperationException(
					"Inequality constraints not supported by minimizeEqualityConstrained");
		}
		return minimizeEqualityConstrainedRaw(fun, grad, hess, x0, eq, nEqOf(eq),
				maxIter, xtol, gtol);
	}

	private static OptimizeResult minimizeEqualityConstrainedRaw(
			java.util.function.Function<Matrix, Double> fun,
			java.util.function.Function<Matrix, Matrix> grad,
			java.util.function.Function<Matrix, Matrix> hess,
			Matrix x0, Object eq, int nEq,
			int maxIter, double xtol, double gtol) {
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
			double f = fun.apply(x);
			Matrix c = (eqConstr == null || nEq == 0) ? emptyC : eqConstr.constrEq(x);
			return new FunctionAndConstraint(f, c);
		};
		IGradientAndJacobian gradAndJac = x -> {
			Matrix g = grad.apply(x);
			Matrix J = (eqJac == null || nEq == 0) ? emptyJ : eqJac.jacEq(x);
			return new GradientAndJacobian(g, J);
		};

		// Lagrangian Hessian: H_objective(x) + sum_i v[i] * H_{c_i}(x).
		// LinearConstraint contributes 0; NonlinearConstraint contributes via its
		// optional hess(x, v) callable. For pure-equality dispatch, v is the
		// equality-multiplier vector with no inequality component.
		final Object eqRef = eq;
		LagrangeHessian lagrHess = (x, v) -> {
			Matrix H = hess.apply(x);
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
			if (optimality < gtol && constrViolation < gtol) return true;
			if (trustRadius < xtol) return true;
			if (s.nIter >= maxIter) return true;
			return false;
		};

		// Run the SQP loop.
		LinearOperator scaling = EqualityConstrainedSQP.defaultScaling(nVars);
		StatefulResult sr = EqualityConstrainedSQP.eqSQP(
				funAndConstr, gradAndJac, lagrHess,
				x0, f0, g0, c0, j0,
				stop, state,
				1.0,                         // initial penalty
				1.0,                         // initial trust radius
				ProjectionMethod.QR_FACTORIZATION,
				null, null,                  // trustLb/Ub: unconstrained
				scaling);

		// Populate OptimizeResult — re-evaluate the objective at the final x to
		// keep r.fun and r.grad consistent with r.x. (ScalarFunction internal
		// state is only tracked for evaluations the SQP routes through it; our
		// closures bypass that.)
		OptimizeResult r = new OptimizeResult();
		r.x = sr.x();
		r.fun = fun.apply(sr.x());
		r.grad = grad.apply(sr.x());
		r.lagrangianGrad = state.lagrangianGrad;
		r.optimality = state.optimality;
		r.constraintViolation = state.constrViolation;
		r.nIter = state.nIter;
		r.numFunctionEval = objective.numFunctionEvals();
		r.numJacobianEval = objective.numGradientEvals();
		r.numHessianEval = objective.numHessianEvals();
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
	 */
	public static <C extends Constraint & Jacobian> OptimizeResult minimize(
			java.util.function.Function<Matrix, Double> fun,
			java.util.function.Function<Matrix, Matrix> grad,
			java.util.function.Function<Matrix, Matrix> hess,
			Matrix x0, C constraint,
			int maxIter, double xtol, double gtol) {
		int nIneq = nIneqOf(constraint);
		if (nIneq == 0) {
			return minimizeEqualityConstrained(fun, grad, hess, x0, constraint, maxIter, xtol, gtol);
		}
		return minimizeInequalityConstrained(fun, grad, hess, x0, constraint, maxIter, xtol, gtol);
	}

	/**
	 * Multi-constraint entry point: combines the supplied constraints into a
	 * {@link CombinedConstraint} and dispatches as in {@link #minimize}. Pass
	 * {@code null} or an empty array for unconstrained problems.
	 */
	public static OptimizeResult minimize(
			java.util.function.Function<Matrix, Double> fun,
			java.util.function.Function<Matrix, Matrix> grad,
			java.util.function.Function<Matrix, Matrix> hess,
			Matrix x0, Object[] constraints,
			int maxIter, double xtol, double gtol) {
		int nVars = (int) x0.getRowCount();
		if (constraints == null || constraints.length == 0) {
			return minimizeEqualityConstrained(fun, grad, hess, x0, null, maxIter, xtol, gtol);
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
					combined.nEq(), maxIter, xtol, gtol);
		}
		return minimizeInequalityConstrainedRaw(fun, grad, hess, x0, combined,
				combined.nEq(), combined.nIneq(), maxIter, xtol, gtol);
	}

	private static int nIneqOf(Object constraint) {
		if (constraint == null) return 0;
		if (constraint instanceof LinearConstraint) return ((LinearConstraint) constraint).nIneq();
		if (constraint instanceof NonlinearConstraint) return ((NonlinearConstraint) constraint).nIneq();
		if (constraint instanceof CombinedConstraint) return ((CombinedConstraint) constraint).nIneq();
		throw new IllegalArgumentException("Unsupported constraint type: " + constraint.getClass().getName());
	}

	/**
	 * Sum of constraint Hessian-of-Lagrangian contributions across the (possibly
	 * combined) constraint. {@link LinearConstraint} contributes 0;
	 * {@link NonlinearConstraint} contributes via its
	 * {@link NonlinearConstraint#lagrangianContribution} callable when set.
	 * Returns {@code null} if no constraint contributes a Hessian — the caller
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
		return null;
	}

	/** Extract a UJMP column vector ({@code n x 1}) as a {@code double[n]}. */
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
			java.util.function.Function<Matrix, Double> fun,
			java.util.function.Function<Matrix, Matrix> grad,
			java.util.function.Function<Matrix, Matrix> hess,
			Matrix x0, C constraint,
			int maxIter, double xtol, double gtol) {
		return minimizeInequalityConstrainedRaw(fun, grad, hess, x0, constraint,
				nEqOf(constraint), nIneqOf(constraint), maxIter, xtol, gtol);
	}

	private static OptimizeResult minimizeInequalityConstrainedRaw(
			java.util.function.Function<Matrix, Double> fun,
			java.util.function.Function<Matrix, Matrix> grad,
			java.util.function.Function<Matrix, Matrix> hess,
			Matrix x0, Object constraint, int nEq, int nIneq,
			int maxIter, double xtol, double gtol) {
		final int nVars = (int) x0.getRowCount();
		final Constraint constr = (Constraint) constraint;
		final Jacobian jac = (Jacobian) constraint;

		// Wrap fun/grad with a no-op args parameter to match the inner method's BiFunction shape.
		final ToDoubleBiFunction<Matrix, Object> funBi = (x, args) -> fun.apply(x);
		final BiFunction<Matrix, Object, Matrix> gradBi = (x, args) -> grad.apply(x);

		// Lagrangian Hessian: H_objective(x) + sum_i v[i] * H_{c_i}(x). The IP path
		// passes v with eq multipliers first and ineq second, so we slice
		// accordingly before delegating to the constraint's Hessian-of-Lagrangian
		// callable (NonlinearConstraint.lagrangianContribution); LinearConstraint
		// contributions are 0 by construction.
		final Object constraintRef = constraint;
		final int nEqLocal = nEq;
		final int nIneqLocal = nIneq;
		LagrangeHessian lagrHess = (x, v) -> {
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
		double f0 = fun.apply(x0);
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
			if (optimality < gtol && constrViolation < gtol) return true;
			if (trustRadius < xtol && barrierParameter < gtol) return true;
			if (s.nIter >= maxIter) return true;
			return false;
		};

		StatefulResult sr = TrustRegionInteriorPoint.trustRegionInteriorPoint(
				funBi, gradBi, lagrHess,
				nVars, nIneq, nEq,
				constr, jac,
				x0, f0, g0,
				cIneq0, jIneq0, cEq0, jEq0,
				stop,
				new boolean[nIneq],
				xtol, state, 0.1, 0.1,
				1.0, 1.0,
				ProjectionMethod.AUGMENTED_SYSTEM);

		// Populate result
		OptimizeResult r = new OptimizeResult();
		r.x = sr.x();
		r.fun = fun.apply(sr.x());
		r.grad = grad.apply(sr.x());
		r.lagrangianGrad = state.lagrangianGrad;
		r.optimality = state.optimality;
		r.constraintViolation = state.constrViolation;
		r.nIter = state.nIter;
		r.cgIter = state.cgNIter;
		r.cgStopCond = state.cgStopCond == null ? PCGStoppingCondition.NOT_EVALUATED : state.cgStopCond;
		r.method = TrustConstrMethod.TRUST_REGION_INTERIOR_POINT;
		r.trustRadius = state.trustRadius;
		r.constraintPenalty = state.constraintPenalty;
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
		return r;
	}

	/**
	 * Minimize a scalar function subject to constraints.
	 *
	 * @see [1] Conn, A. R., Gould, N. I., & Toint, P. L.
     *          Trust region methods. 2000. Siam. pp. 19.
	 *
	 * @param fun
	 * @param x0
	 * @param args
	 * @param grad
	 * @param hess
	 * @param hessp
	 * @param bounds
	 * @param xTol                     Tolerance for termination by the change of
	 *                                 the independent variable. The algorithm will
	 *                                 terminate when ``tr_radius < xtol``, where
	 *                                 ``tr_radius`` is the radius of the trust
	 *                                 region used in the algorithm. Default is
	 *                                 1e-8.
	 * @param gTol                     Tolerance for termination by the norm of the
	 *                                 Lagrangian gradient. The algorithm will
	 *                                 terminate when both the infinity norm (i.e.,
	 *                                 max abs value) of the Lagrangian gradient and
	 *                                 the constraint violation are smaller than
	 *                                 ``gtol``. Default is 1e-8.
	 * @param barrierTol               Threshold on the barrier parameter for the
	 *                                 algorithm termination. When inequality
	 *                                 constraints are present, the algorithm will
	 *                                 terminate only when the barrier parameter is
	 *                                 less than `barrier_tol`. Default is 1e-8.
	 * @param sparseJacobian           Determines how to represent Jacobians of the
	 *                                 constraints. If bool, then Jacobians of all
	 *                                 the constraints will be converted to the
	 *                                 corresponding format. If None (default), then
	 *                                 Jacobians won't be converted, but the
	 *                                 algorithm can proceed only if they all have
	 *                                 the same format.
	 * @param callback
	 * @param maxIter                  Maximum number of algorithm iterations.
	 *                                 Default is 1000.
	 * @param verbose                  Level of algorithm's verbosity:
	 *
	 *                                 0 (default) : work silently. 1 : display a
	 *                                 termination report. 2 : display progress
	 *                                 during iterations. 3 : display progress
	 *                                 during iterations (more complete report).
	 * @param finiteDifferenceRelStep  Relative step size for the finite difference
	 *                                 approximation.
	 * @param initialConstraintPenalty Initial constraints penalty parameter. The
	 *                                 penalty parameter is used for balancing the
	 *                                 requirements of decreasing the objective
	 *                                 function and satisfying the constraints. It
	 *                                 is used for defining the merit function:
	 *                                 ``merit_function(x) = fun(x) + constr_penalty
	 *                                 * constr_norm_l2(x)``, where
	 *                                 ``constr_norm_l2(x)`` is the l2 norm of a
	 *                                 vector containing all the constraints. The
	 *                                 merit function is used for accepting or
	 *                                 rejecting trial points and ``constr_penalty``
	 *                                 weights the two conflicting goals of reducing
	 *                                 objective function and constraints. The
	 *                                 penalty is automatically updated throughout
	 *                                 the optimization process, with
	 *                                 ``initial_constr_penalty`` being its initial
	 *                                 value. Default is 1 (recommended in [1]_, p
	 *                                 19).
	 * @param initialTrustRadius       Initial trust radius. The trust radius gives
	 *                                 the maximum distance between solution points
	 *                                 in consecutive iterations. It reflects the
	 *                                 trust the algorithm puts in the local
	 *                                 approximation of the optimization problem.
	 *                                 For an accurate local approximation the
	 *                                 trust-region should be large and for an
	 *                                 approximation valid only close to the current
	 *                                 point it should be a small one. The trust
	 *                                 radius is automatically updated throughout
	 *                                 the optimization process, with
	 *                                 ``initial_tr_radius`` being its initial
	 *                                 value. Default is 1 (recommended in [1]_, p.
	 *                                 19).
	 * @param initialBarrierParameter  Initial barrier parameter and initial
	 *                                 tolerance for the barrier subproblem. Both
	 *                                 are used only when inequality constraints are
	 *                                 present. For dealing with optimization
	 *                                 problems ``min_x f(x)`` subject to inequality
	 *                                 constraints ``c(x) <= 0`` the algorithm
	 *                                 introduces slack variables, solving the
	 *                                 problem ``min_(x,s) f(x) +
	 *                                 barrier_parameter*sum(ln(s))`` subject to the
	 *                                 equality constraints ``c(x) + s = 0`` instead
	 *                                 of the original problem. This subproblem is
	 *                                 solved for decreasing values of
	 *                                 ``barrier_parameter`` and with decreasing
	 *                                 tolerances for the termination, starting with
	 *                                 ``initial_barrier_parameter`` for the barrier
	 *                                 parameter and ``initial_barrier_tolerance``
	 *                                 for the barrier tolerance. Default is 0.1 for
	 *                                 both values (recommended in [1]_ p. 19). Also
	 *                                 note that ``barrier_parameter`` and
	 *                                 ``barrier_tolerance`` are updated with the
	 *                                 same prefactor.
	 * @param initialBarrierTolerance
	 * @param factorizationMethod      Method to factorize the Jacobian of the
	 *                                 constraints. Use None (default) for the auto
	 *                                 selection or one of:
	 *
	 *                                 - 'NormalEquation' (requires scikit-sparse) -
	 *                                 'AugmentedSystem' - 'QRFactorization' -
	 *                                 'SVDFactorization'
	 *
	 *                                 The methods 'NormalEquation' and
	 *                                 'AugmentedSystem' can be used only with
	 *                                 sparse constraints. The projections required
	 *                                 by the algorithm will be computed using,
	 *                                 respectively, the the normal equation and the
	 *                                 augmented system approaches explained in
	 *                                 [1]_. 'NormalEquation' computes the Cholesky
	 *                                 factorization of ``A A.T`` and
	 *                                 'AugmentedSystem' performs the LU
	 *                                 factorization of an augmented system. They
	 *                                 usually provide similar results.
	 *                                 'AugmentedSystem' is used by default for
	 *                                 sparse matrices.
	 *
	 *                                 The methods 'QRFactorization' and
	 *                                 'SVDFactorization' can be used only with
	 *                                 dense constraints. They compute the required
	 *                                 projections using, respectively, QR and SVD
	 *                                 factorizations. The 'SVDFactorization' method
	 *                                 can cope with Jacobian matrices with
	 *                                 deficient row rank and will be used whenever
	 *                                 other factorization methods fail (which may
	 *                                 imply the conversion of sparse matrices to a
	 *                                 dense format when required). By default,
	 *                                 'QRFactorization' is used for dense matrices.
	 * @param disp                     If True (default), then `verbose` will be set
	 *                                 to 1 if it was 0.
	 */
	public static OptimizeResult minimizeTrustConstr(ToDoubleBiFunction<Matrix, Object> fun, Matrix x0, Object args,
			BiFunction<Matrix, Object, Matrix> grad, BiFunction<Matrix, Object, Matrix> hess,
			HessianProduct hessp, Bounds bounds,
			Object constraints,
			double xTol, double gTol, double barrierTol, Optional<Boolean> sparseJacobian,
			Object callback, int maxIter, int verbose, Matrix finiteDifferenceRelStep, double initialConstraintPenalty,
			double initialTrustRadius, double initialBarrierParameter, double initialBarrierTolerance,
			ProjectionMethod factorizationMethod, boolean disp) {

		OptimizeResult result = new OptimizeResult();

		long nVars = x0.getRowCount();

		if (hess == null) {
			if (hessp != null) {
				hess = new HessianLinearOperator(hessp, nVars);
			} else {
				hess = BFGS.FACTORY.build();
			}
		}

		if (disp && verbose == 0) {
			verbose = 1;
		}

		FiniteDifferenceBounds finiteDiffBounds;
		if (bounds != null) {
			finiteDiffBounds = new StrictBounds(bounds);
		} else {
			finiteDiffBounds = FiniteDifferenceBounds.unbounded(nVars);
		}

		// Define Objective Function
		ScalarFunction objective = ScalarFunction.FACTORY
				.fun(fun)
				.x0(x0)
				.args(args)
				.grad(grad)
				.hess(hess)
				.finiteDiffRelStep(finiteDifferenceRelStep)
				.finiteDiffBounds(finiteDiffBounds)
				.build();

		// Put constraints in list format when needed.
		Constraint[] rawConstraints;
		if (constraints instanceof NonlinearConstraint || constraints instanceof LinearConstraint) {
			rawConstraints = new Constraint[] {(Constraint) constraints};
		} else {
			rawConstraints = (Constraint[]) constraints;
		}

		// Prepare constraints.
		List<PreparedConstraint> preparedConstraints = new LinkedList<>();
		for (Constraint c: rawConstraints) {
			preparedConstraints.add(new PreparedConstraint(c, x0, sparseJacobian, finiteDiffBounds));
		}

		// Check that all constraints are either sparse or dense.
		int nSparse = 0;
		for (PreparedConstraint pc: preparedConstraints) {
			if (pc.fun().sparseJacobian()) {
				nSparse++;
			}
		}

		if (nSparse > 0 && nSparse < preparedConstraints.size()) {
			throw new RuntimeException("All constraints must have the same kind of the \n" +
					"Jacobian --- either all sparse or all dense. \n" +
					"You can set the sparsity globally by setting \n" +
					"`sparse_jacobian` to either True of False.");
		}

		if (preparedConstraints != null) {
			sparseJacobian = Optional.of(nSparse > 0);
		}

		if (bounds != null) {
			if (!sparseJacobian.isPresent()) {
				sparseJacobian = Optional.of(true);
			}
			preparedConstraints.add(new PreparedConstraint(bounds, x0, sparseJacobian));
		}

		// Concatenate initial constraints to the canonical form.




		// Prepare all canonical constraints and concatenate it into one.









		// Generate the Hessian of the Lagrangian.





		// Choose appropriate method






		// Construct OptimizeResult





		// Start counting
		long startTime = System.nanoTime();


		// Define stop criteria





		// Call inferior function to do the optimization




		// Status 3 occurs when the callback function requests termination,
	    // this is assumed to not be a success.







		return result;
	}

}
