package org.scipy.optimize.minimize;

import java.util.LinkedList;
import java.util.List;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.ToDoubleBiFunction;

import org.scipy.optimize.minimize.enums.ProjectionMethod;
import org.scipy.optimize.minimize.interfaces.Constraint;
import org.scipy.optimize.minimize.records.Bounds;
import org.scipy.optimize.minimize.records.FiniteDifferenceBounds;
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
