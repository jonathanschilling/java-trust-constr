package org.scipy.optimize.minimize;

import java.util.function.BiFunction;
import java.util.function.ToDoubleBiFunction;

import org.scipy.optimize.minimize.enums.ProjectionMethod;
import org.scipy.optimize.minimize.interfaces.Constraint;
import org.scipy.optimize.minimize.interfaces.GlobalStoppingCriteria;
import org.scipy.optimize.minimize.interfaces.Jacobian;
import org.scipy.optimize.minimize.interfaces.LagrangeHessian;
import org.scipy.optimize.minimize.records.FunctionAndConstraint;
import org.scipy.optimize.minimize.records.GradientAndJacobian;
import org.scipy.optimize.minimize.records.State;
import org.scipy.optimize.minimize.records.StatefulResult;
import org.scipy.optimize.minimize.matrix.Matrix;

/**
 * Trust-region interior-point method.
 *
 * <p>References:
 * <ol>
 *   <li>Byrd, Hribar, Nocedal, &quot;An interior point algorithm for
 *       large-scale nonlinear programming&quot;, SIAM J. Optim. 9.4
 *       (1999): 877-900.</li>
 *   <li>Byrd, Liu, Nocedal, &quot;On the local behavior of an interior
 *       point method for nonlinear programming&quot;, Numerical Analysis
 *       1997 (1997): 37-56.</li>
 *   <li>Nocedal &amp; Wright, <i>Numerical Optimization</i>, 2nd ed.
 *       (2006).</li>
 * </ol>
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
	 * Trust-region interior-point solve.
	 *
	 * <p>Minimizes
	 * <pre>
	 *   minimize    fun(x)
	 *   subject to  constr_ineq(x) &lt;= 0
	 *               constr_eq(x)   =  0
	 * </pre>
	 * using the algorithm of Byrd-Hribar-Nocedal (1999).
	 *
	 * @param fun      objective {@code (x, args) -> f(x)}
	 * @param grad     gradient {@code (x, args) -> gradf(x)}
	 * @param lagrHess Hessian-of-Lagrangian {@code (x, vEq, vIneq) -> grad^2L(x, v)}
	 * @param nVars    number of decision variables
	 * @param nIneq    number of canonical inequality rows ({@code c(x) <= 0})
	 * @param nEq      number of canonical equality rows ({@code c(x) = 0})
	 * @param constr   combined-constraint evaluator that returns {@code (eq, ineq)}
	 * @param jac      combined-constraint Jacobian evaluator
	 * @param x0       starting iterate ({@code n x 1})
	 * @param fun0     {@code fun(x0)}
	 * @param grad0    {@code grad(x0)}
	 * @param constrIneq0 {@code constr_ineq(x0)} ({@code nIneq x 1})
	 * @param jacIneq0    inequality Jacobian at {@code x0} ({@code nIneq x n})
	 * @param constrEq0   {@code constr_eq(x0)} ({@code nEq x 1})
	 * @param jacEq0      equality Jacobian at {@code x0} ({@code nEq x n})
	 * @param stopCrit termination criterion (queried each outer iteration)
	 * @param enforceFeasibility per-canonical-ineq flag forcing strict feasibility
	 *                           in the slack rows; may be {@code null} (no
	 *                           enforcement)
	 * @param xtol     termination tolerance on {@code trustRadius}
	 * @param state    mutable iteration state to thread through and return
	 * @param initialBarrierParameter initial value of the log-barrier coefficient
	 * @param initialTolerance initial inner-loop tolerance for the barrier subproblem
	 * @param initialPenalty initial constraint penalty for the merit function
	 * @param initialTrustRadius initial trust-region radius
	 * @param factorizationMethod projection-Jacobian factorization method
	 *                            ({@code AUGMENTED_SYSTEM} / {@code QR_FACTORIZATION}
	 *                            / {@code SVD_FACTORIZATION})
	 * @return populated {@link StatefulResult}
	 */
	public static StatefulResult trustRegionInteriorPoint(
			ToDoubleBiFunction<Matrix, Object> fun, BiFunction<Matrix, Object, Matrix> grad, LagrangeHessian lagrHess,
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
		for (int i = 0; i < nIneq; ++i) {
			s0.setAsDouble(Math.max(-1.5 * constrIneq0.getAsDouble(i, 0), 1.0), i, 0);
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
		trustUb.fill(Double.POSITIVE_INFINITY);

		// Solve a sequence of barrier problems
		Matrix vFinal = null;
		while(true) {

			// Solve SQP subproblem
			StatefulResult r = EqualityConstrainedSQP.eqSQP(subProb::funAndConstr,
					subProb::gradAndJac, subProb::lagrangianHessian,
					z, fun0SubProb, grad0SubProb, constr0SubProb, jac0SubProb,
					subProb::stoppingCriteria, state, initialPenalty, trustRadius,
					factorizationMethod, trustLb, trustUb, subProb::getScaling);
			z = r.x();
			state = r.state();
			vFinal = r.v();

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
			fun0SubProb = fc.f();
			constr0SubProb = fc.c();

			GradientAndJacobian gj = subProb.gradAndJac(z);
			grad0SubProb = gj.grad();
			jac0SubProb = gj.jac();
		}

		// Get x and s
		Matrix x = subProb.getVariables(z);

		// vFinal is the augmented-system multiplier from the final barrier
		// subproblem: length nEq + nIneq; first nEq entries are equality
		// multipliers, remaining nIneq entries are slack-row multipliers
		// corresponding to the original problem's inequality lambda.
		return new StatefulResult(x, state, vFinal);
	}
}
