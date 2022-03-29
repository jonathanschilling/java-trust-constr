package org.scipy.optimize.minimize;

import org.scipy.optimize.minimize.enums.PCGStoppingCondition;
import org.scipy.optimize.minimize.enums.ProjectionMethod;
import org.scipy.optimize.minimize.interfaces.IFunctionAndConstraint;
import org.scipy.optimize.minimize.interfaces.IGradientAndJacobian;
import org.scipy.optimize.minimize.interfaces.LagrangeHessian;
import org.scipy.optimize.minimize.interfaces.LinearOperator;
import org.scipy.optimize.minimize.interfaces.StoppingCriterion;
import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

/** Byrd-Omojokun Trust-Region SQP method */
public class EqualityConstrainedSQP {

	/** Rho from formula (3.51), reference [2], p.891 */
	static final double PENALTY_FACTOR = 0.3;

	static final double LARGE_REDUCTION_RATIO = 0.9;

	static final double INTERMEDIARY_REDUCTION_RATIO = 0.3;

	/** Eta from reference [2], p.892 */
	static final double SUFFICIENT_REDUCTION_RATIO = 1e-8;

	static final double TRUST_ENLARGEMENT_FACTOR_L = 7.0;

	static final double TRUST_ENLARGEMENT_FACTOR_S = 2.0;

	static final double MAX_TRUST_REDUCTION = 0.5;

	static final double MIN_TRUST_REDUCTION = 0.1;

	static final double SOC_THRESHOLD = 0.1;

	/** Zeta from formula (3.21), reference [2], p.885 */
	static final double TR_FACTOR = 0.8;

	static final double BOX_FACTOR = 0.5;

	public static final LinearOperator defaultScaling(long n) {
		return new LinearOperator() {
			@Override
			public Matrix apply(Matrix x) {
				// default: no scaling
				return Matrix.Factory.copyFromMatrix(x);
			}
		};
	}

	/**
	 * Solve nonlinear equality-constrained problem using trust-region SQP.
	 * </p>
	 * Solve optimization problem:
	 * <pre>
	 * minimize fun(x)
	 * subject to: constr(x) = 0
	 * </pre>
	 * using Byrd-Omojokun Trust-Region SQP method described in [1].
	 * Several implementation details are based on [2] and [3], p. 549.
	 *
	 * @see [1] Lalee, Marucha, Jorge Nocedal, and Todd Plantenga
	 *          "On the implementation of an algorithm for large-scale equality constrained optimization"
	 *          SIAM Journal on Optimization 8.3 (1998), p. 682-706
	 * @see [2] Byrd, Richard H., Mary E. Hribar, and Jorge Nocedal
	 *          "An interior point algorithm for large-scale nonlinear programming"
	 *          SIAM Journal on Optimization 9.4 (1999): 877-900
	 * @see [3] Nocedal, Jorge, and Stephen J. Wright
	 *          "Numerical optimization", Second Edition (2006)
	 *
	 */
	public static StatefulResult eqSQP(
			IFunctionAndConstraint funAndConstr,
			IGradientAndJacobian gradAndJac,
			LagrangeHessian lagrHess,
			Matrix x0, double fun0, Matrix grad0, Matrix constr0, Matrix jac0,
			StoppingCriterion stopCrit,
			State state,
			double initialPenalty, double initialTrustRadius,
			ProjectionMethod factorizationMethod,
			Matrix trustLb, Matrix trustUb, LinearOperator scaling) {

		// Number of parameters
		final long n = x0.getRowCount();

		// Set default lower and upper bounds.
		if (trustLb == null) {
			trustLb = Matrix.Factory.zeros(n, 1).fill(Ret.ORIG, Double.NEGATIVE_INFINITY);
		}
		if (trustUb == null) {
			trustUb = Matrix.Factory.zeros(n, 1).fill(Ret.ORIG, Double.NEGATIVE_INFINITY);
		}

		// Initial values
		Matrix x = Matrix.Factory.copyFromMatrix(x0);
		double trustRadius = initialTrustRadius;
		double penalty = initialPenalty;

		// Compute Values
		double f = fun0;
		Matrix c = Matrix.Factory.copyFromMatrix(grad0);
		Matrix b = Matrix.Factory.copyFromMatrix(constr0);
		Matrix A = Matrix.Factory.copyFromMatrix(jac0);
		Matrix S = scaling.apply(x);

		// Get projections
		LinearOperator[] op = Projections.projections(A, factorizationMethod);
		LinearOperator Z = op[0]; // nullspace
		LinearOperator LS = op[1]; // least-squares
		LinearOperator Y = op[2]; // rowspace

		// Compute least-square lagrange multipliers
		Matrix v = LS.apply(c).times(-1);

		// Compute Hessian
		LinearOperator H = lagrHess.lagrHess(x, v);

		// Update state parameters
		double optimality = c.plus(A.transpose().mtimes(v)).normInf();
		double constrViolation;
		if (b.getRowCount() > 0) {
			constrViolation = b.normInf();
		} else {
			constrViolation = 0.0;
		}
		CGInfo cgInfo = new CGInfo();
		cgInfo.niter = 0;
		cgInfo.stopCond = PCGStoppingCondition.NOT_EVALUATED;
		cgInfo.hitsBoundary = false;

		boolean lastIterationFailed = false;
		while (!stopCrit.shouldStop(
				state, x,
				lastIterationFailed,
				optimality, constrViolation,
				trustRadius, penalty, cgInfo)) {

			// Normal Step - `dn`
			// minimize 1/2*||A dn + b||^2
			// subject to:
			// ||dn|| <= TR_FACTOR * trust_radius
			// BOX_FACTOR * lb <= dn <= BOX_FACTOR * ub.
			Matrix dn = QPSubproblem.modifiedDogleg(A, Y, b,
					TR_FACTOR*trustRadius,
					LinAlg.col(trustLb.times(BOX_FACTOR)),
					LinAlg.col(trustUb.times(BOX_FACTOR)));

			// Tangential Step - `dt`
			// Solve the QP problem:
			// minimize 1/2 dt.T H dt + dt.T (H dn + c)
			// subject to:
			// A dt = 0
			// ||dt|| <= sqrt(trust_radius**2 - ||dn||**2)
			// lb - dn <= dt <= ub - dn
			Matrix c_t = H.apply(dn).plus(c);
			Matrix b_t = Matrix.Factory.zeros(b.getRowCount(), b.getColumnCount());
			double normDn = dn.norm2();
			double trustRadiusT = Math.sqrt(trustRadius*trustRadius - normDn*normDn);
			Matrix lbT = trustLb.minus(dn);
			Matrix ubT = trustUb.minus(dn);
			CGInfo dtResult = QPSubproblem.projectedCG(H, c_t, Z, Y, b_t,
					trustRadiusT, LinAlg.col(lbT), LinAlg.col(ubT), Double.NaN); // default tolerance
			Matrix dt = dtResult.x;

			// Compute update (normal + tangential steps).
			Matrix d = dn.plus(dt);

			// Compute second order model: 1/2 d H d + c.T d + f
			double quadraticModel = 0.5 * d.transpose().mtimes(H.apply(d)).doubleValue() + c.transpose().mtimes(c).doubleValue();

			// Compute linearized constraint: l = A d + b.
			Matrix linearizedConstr = A.mtimes(d).plus(b);

			// Compute new penalty parameter according to formula (3.52),
			// reference [2], p.891.
			double vPred = b.norm2() - linearizedConstr.norm2();

			// Guarantee `vpred` always positive, regardless of roundoff errors.
			vPred = Math.max(1.0e-16, vPred);

			double previousPenalty = penalty;
			if (quadraticModel > 0) {
				double newPenalty = quadraticModel / ((1.0 - PENALTY_FACTOR) * vPred);
				penalty = Math.max(penalty, newPenalty);
			}

			// Compute predicted reduction according to formula (3.52), reference [2], p.891.
			double predictedReduction = -quadraticModel + penalty*vPred;

			// Compute merit function at current point
			double meritFunction = f + penalty * b.norm2();

			// Evaluate function and constraints at trial point
			Matrix xNext = x.plus(S.mtimes(d));
			FunctionAndConstraint fc = funAndConstr.funAndConstr(xNext);
			double fNext = fc.f();
			Matrix bNext = fc.c();

			// Compute merit function at trial point
			double meritFunctionNext = fNext + penalty * bNext.norm2();

			// Compute actual reduction according to formula (3.54), reference [2], p.892.
			double actualReduction = meritFunction - meritFunctionNext;

			// Compute reduction ratio
			double reductionRatio = actualReduction / predictedReduction;

			// Second order correction (SOC), reference [2], p.892.
			if (reductionRatio < SUFFICIENT_REDUCTION_RATIO && dn.norm2() <= SOC_THRESHOLD * dt.norm2()) {

				// Compute second order correction
				Matrix y = Y.apply(bNext).times(-1);

				// Make sure increment is inside box constraints
				IntersectionResult r = QPSubproblem.boxIntersections(
						LinAlg.col(d), LinAlg.col(y), LinAlg.col(trustLb), LinAlg.col(trustUb));
				double t = r.tB();

				// Compute tentative point
				Matrix xSoc = x.plus(S.mtimes(d.plus(y.times(t))));
				FunctionAndConstraint fcSoc = funAndConstr.funAndConstr(xSoc);
				double fSoc = fcSoc.f();
				Matrix bSoc = fcSoc.c();

				// Recompute actual reduction
				double meritFunctionSoc = fSoc + penalty * bSoc.norm2();
				double actualReductionSoc = meritFunction - meritFunctionSoc;

				// Recompute reduction ratio
				double reductionRatioSoc = actualReductionSoc / predictedReduction;
				if (r.intersect() && reductionRatioSoc >= SUFFICIENT_REDUCTION_RATIO) {
					xNext = xSoc;
					fNext = fSoc;
					bNext = bSoc;
					reductionRatio = reductionRatioSoc;
				}
			}

			// Readjust trust region step, formula (3.55), reference [2], p.892.
			if (reductionRatio >= LARGE_REDUCTION_RATIO) {
				trustRadius = Math.max(TRUST_ENLARGEMENT_FACTOR_L * d.norm2(), trustRadius);
			} else if (reductionRatio >= INTERMEDIARY_REDUCTION_RATIO) {
				trustRadius = Math.max(TRUST_ENLARGEMENT_FACTOR_S * d.norm2(), trustRadius);
			} else if (reductionRatio < SUFFICIENT_REDUCTION_RATIO) {
				// Reduce trust region step, according to reference [3], p.696.
				double trustReduction = (1.0 - SUFFICIENT_REDUCTION_RATIO) / (1.0 - reductionRatio);
				double newTrustRadius = trustReduction * d.norm2();
				if (newTrustRadius >= MAX_TRUST_REDUCTION * trustRadius) {
					trustRadius *= MAX_TRUST_REDUCTION;
				} else if (newTrustRadius >= MIN_TRUST_REDUCTION * trustRadius) {
					trustRadius = newTrustRadius;
				} else {
					trustRadius *= MIN_TRUST_REDUCTION;
				}
			}

			// Update iteration
			if (reductionRatio >= SUFFICIENT_REDUCTION_RATIO) {
				x = xNext;
				f = fNext;
				b = bNext;
				GradientAndJacobian gj = gradAndJac.gradAndJac(x);
				c = gj.grad();
				A = gj.jac();
				S = scaling.apply(x);

				// Get projections
				op = Projections.projections(A, factorizationMethod);
				Z = op[0]; // nullspace
				LS = op[1]; // least-squares
				Y = op[2]; // rowspace

				// Compute least-square lagrange multipliers
				v = LS.apply(c).times(-1);

				// Compute Hessian
				H = lagrHess.lagrHess(x, v);

				// Set Flag
				lastIterationFailed = false;

				// Optimality values
				optimality = c.plus(A.transpose().mtimes(v)).normInf();
				if (b.getRowCount() > 0) {
					constrViolation = b.normInf();
				} else {
					constrViolation = 0.0;
				}
			} else {
				penalty = previousPenalty;
				lastIterationFailed = true;
			}
		}

		return new StatefulResult(x, state);
	}
}
