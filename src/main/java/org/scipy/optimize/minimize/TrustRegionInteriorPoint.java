package org.scipy.optimize.minimize;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

/**
 * Trust-region interior point method.
 *
 * @see [1] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal.
 *          "An interior point algorithm for large-scale nonlinear
 *          programming." SIAM Journal on Optimization 9.4 (1999): 877-900.
 * @see [2] Byrd, Richard H., Guanghui Liu, and Jorge Nocedal.
 *          "On the local behavior of an interior point method for
 *          nonlinear programming." Numerical analysis 1997 (1997): 37-56.
 * @see [3] Nocedal, Jorge, and Stephen J. Wright. "Numerical optimization"
 *          Second Edition (2006).
 */
public class TrustRegionInteriorPoint {

	/**
	 * BOUNDARY_PARAMETER controls the decrease on the slack variables.
	 * Represents {@code tau} from [1], p.885, formula (3.18).
	 */
	static final double BOUNDARY_PARAMETER = 0.995;

	/**
	 * BARRIER_DECAY_RATIO controls the decay of the barrier parameter and of the subproblem tolerance.
	 * Represents {@code theta} from [1], p.879.
	 */
	static final double BARRIER_DECAY_RATIO = 0.2;

	/** TRUST_ENLARGEMENT controls the enlargement on trust radius after each iteration. */
	static final double TRUST_ENLARGEMENT = 5.0;

	/**
	 * Trust-region interior points method.
	 *
	 * Solve problem:
	 * <pre>
	 * minimize fun(x)
	 * subject to: constr_ineq(x) <= 0
	 *             constr_eq(x) = 0
	 * </pre>
	 * using trust-region interior point method described in [1].
	 */
	public static StatefulResult trustRegionInteriorPoint(
			Function fun, Gradient grad, LagrangeHessian lagrHess,
			int nVars, int nIneq, int nEq,
			Constraint constr, Jacobian jac,
			Matrix x0, double fun0, Matrix grad0,
			Matrix constrIneq0, Matrix jacIneq0,
			Matrix constrEq0, Matrix jacEq0,
			GlobalStoppingCriteria stopCrit,
			boolean[] enforceFeasibility,
			double xtol, State state, double initialBarrierParameter,
			double initialTolerance, double initialPenalty,
			double initialTrustRadius, ProjectionMethod factorizationMethod) {

		// Default enforce_feasibility
		if (enforceFeasibility == null) {
			enforceFeasibility = new boolean[nIneq];
		}

		// Initial Values
		double barrierParameter = initialBarrierParameter;
		double tolerance = initialTolerance;
		double trustRadius = initialTrustRadius;

		// Define initial value for the slack variables
		Matrix s0 = Matrix.Factory.zeros(nIneq, 1);
		for (long[] pos: constrIneq0.allCoordinates()) {
			s0.setAsDouble(Math.max(-1.5 * constrIneq0.getAsDouble(pos), 1.0), pos);
		}

		// Define barrier subproblem
		BarrierSubproblem subProb = new BarrierSubproblem(x0, s0,
				fun, grad, lagrHess,
				nVars, nIneq, nEq,
				constr, jac,
				barrierParameter, tolerance, enforceFeasibility, stopCrit,
				xtol, fun0, grad0, constrIneq0, jacIneq0, constrEq0, jacEq0);

		// Define initial parameter for the first iteration.
		Matrix z = Matrix.Factory.vertCat(x0, s0);
		double fun0SubProb = subProb.fun0;
		Matrix constr0SubProb = subProb.constr0;
		Matrix grad0SubProb = subProb.grad0;
		Matrix jac0SubProb = subProb.jac0;

		// Define trust region bounds
		Matrix lbVar  = Matrix.Factory.fill(Double.NEGATIVE_INFINITY, subProb.nVars, 1);
		Matrix lbIneq = Matrix.Factory.fill(-BOUNDARY_PARAMETER, subProb.nIneq, 1);
		Matrix trustLb = Matrix.Factory.vertCat(lbVar, lbIneq);

		Matrix trustUb = Matrix.Factory.zeros(subProb.nVars + subProb.nIneq, 1);
		trustUb.fill(Ret.ORIG, Double.POSITIVE_INFINITY);

		// Solve a sequence of barrier problems
		while(true) {

			// Solve SQP subproblem
			StatefulResult r = EqualityConstrainedSQP.eqSQP(subProb::funAndConstr,
					subProb::gradAndJac, subProb::lagrangianHessian,
					z, fun0SubProb, grad0SubProb, constr0SubProb, jac0SubProb,
					subProb::stoppingCriteria, state, initialPenalty, trustRadius,
					factorizationMethod, trustLb, trustUb, subProb::getScaling);
			z = r.x;
			state = r.state;

			if (subProb.terminate) {
				break;
			}

			// Update parameters
			trustRadius = Math.max(initialTrustRadius, TRUST_ENLARGEMENT * state.trustRadius);

			// TODO: Use more advanced strategies from [2] to update this parameters.
			barrierParameter *= BARRIER_DECAY_RATIO;
			tolerance *= BARRIER_DECAY_RATIO;

			// Update Barrier Problem
			subProb.update(barrierParameter, tolerance);

			// Compute initial values for next iteration
			FunctionAndConstraint fc = subProb.funAndConstr(z);
			fun0SubProb = fc.f;
			constr0SubProb = fc.c;

			GradientAndJacobian gj = subProb.gradAndJac(z);
			grad0SubProb = gj.grad;
			jac0SubProb = gj.jac;
		}

		// Get x and s
		Matrix x = subProb.getVariables(z);

		StatefulResult r = new StatefulResult();
		r.x = x;
		r.state = state;

		return r;
	}
}
