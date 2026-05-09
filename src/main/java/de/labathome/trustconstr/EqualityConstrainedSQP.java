package de.labathome.trustconstr;

import de.labathome.trustconstr.enums.PCGStoppingCondition;
import de.labathome.trustconstr.enums.ProjectionMethod;
import de.labathome.trustconstr.interfaces.IFunctionAndConstraint;
import de.labathome.trustconstr.interfaces.IGradientAndJacobian;
import de.labathome.trustconstr.interfaces.LagrangeHessian;
import de.labathome.trustconstr.interfaces.LinearOperator;
import de.labathome.trustconstr.interfaces.StoppingCriterion;
import de.labathome.trustconstr.records.CGInfo;
import de.labathome.trustconstr.records.FunctionAndConstraint;
import de.labathome.trustconstr.records.GradientAndJacobian;
import de.labathome.trustconstr.records.IntersectionResult;
import de.labathome.trustconstr.records.State;
import de.labathome.trustconstr.records.StatefulResult;
import de.labathome.trustconstr.matrix.Matrix;

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

	/**
	 * Like {@link Matrix#norm2()} but safe for zero-row matrices, which arise on
	 * the unconstrained dispatch path (no equality constraints). The natural
	 * answer is 0; this guard avoids divisions by zero in the merit-function
	 * arithmetic when the constraint Jacobian has shape 0&times;n.
	 */
	private static double safeNorm2(Matrix m) {
		return (m.getRowCount() == 0 || m.getColumnCount() == 0) ? 0.0 : m.norm2();
	}

	/**
	 * Default scaling: returns the {@code n x n} identity, i.e. an unscaled
	 * SQP step. Mirrors scipy's {@code default_scaling}, which returns
	 * {@code scipy.sparse.eye(n)}.
	 *
	 * @param n number of decision variables
	 * @return a {@link LinearOperator} that ignores its input and returns the identity
	 */
	public static final LinearOperator defaultScaling(long n) {
		// "No scaling" means S = I_n: the SQP step is applied unscaled
		// (S.mtimes(d) == d).
		final Matrix identity = Matrix.Factory.eye(n, n);
		return new LinearOperator() {
			@Override
			public Matrix apply(Matrix x) {
				return identity;
			}
		};
	}

	/**
	 * Solve a nonlinear equality-constrained problem using trust-region SQP.
	 *
	 * <p>Minimizes
	 * <pre>
	 *   minimize    fun(x)
	 *   subject to  constr(x) = 0
	 * </pre>
	 * using the Byrd-Omojokun trust-region SQP method (Lalee, Nocedal,
	 * Plantenga, 1998). Implementation details follow Byrd-Hribar-Nocedal
	 * (1999) and Nocedal &amp; Wright, <i>Numerical Optimization</i>,
	 * 2nd ed. (2006), p.549.
	 *
	 * @param funAndConstr combined evaluator returning {@code (f(x), constr(x))}
	 * @param gradAndJac   combined evaluator returning {@code (gradf(x), gradconstr(x))}
	 * @param lagrHess     Hessian-of-Lagrangian {@code (x, v) -> grad^2L(x, v)}
	 * @param x0           starting iterate ({@code n x 1})
	 * @param fun0         {@code fun(x0)}
	 * @param grad0        {@code grad(x0)}
	 * @param constr0      {@code constr(x0)} (m x 1)
	 * @param jac0         constraint Jacobian at {@code x0} (m x n)
	 * @param stopCrit     termination predicate, queried each outer iteration
	 * @param state        mutable iteration state to thread through and return
	 * @param initialPenalty     initial constraint penalty for the merit function
	 * @param initialTrustRadius initial trust-region radius
	 * @param factorizationMethod projection-Jacobian factorization method
	 *                            ({@code AUGMENTED_SYSTEM} / {@code QR_FACTORIZATION}
	 *                            / {@code SVD_FACTORIZATION})
	 * @param trustLb      element-wise lower bounds on the SQP step
	 * @param trustUb      element-wise upper bounds on the SQP step
	 * @param scaling      diagonal-scaling operator for the trust-region step
	 *                     ({@link #defaultScaling(long)} for the identity)
	 * @return populated {@link StatefulResult}
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
			trustLb = Matrix.Factory.zeros(n, 1).fill(Double.NEGATIVE_INFINITY);
		}
		if (trustUb == null) {
			trustUb = Matrix.Factory.zeros(n, 1).fill(Double.POSITIVE_INFINITY);
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
					trustLb.times(BOX_FACTOR).toColumnArray(),
					trustUb.times(BOX_FACTOR).toColumnArray());

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
					trustRadiusT, lbT.toColumnArray(), ubT.toColumnArray(), Double.NaN); // default tolerance
			Matrix dt = dtResult.x;

			// Compute update (normal + tangential steps).
			Matrix d = dn.plus(dt);

			// Compute second order model: 1/2 d H d + c.T d
			double quadraticModel = 0.5 * d.transpose().mtimes(H.apply(d)).doubleValue() + c.transpose().mtimes(d).doubleValue();

			// Compute linearized constraint: l = A d + b.
			Matrix linearizedConstr = A.mtimes(d).plus(b);

			// Compute new penalty parameter according to formula (3.52),
			// reference [2], p.891.
			double vPred = safeNorm2(b) - safeNorm2(linearizedConstr);

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
			double meritFunction = f + penalty * safeNorm2(b);

			// Evaluate function and constraints at trial point
			Matrix xNext = x.plus(S.mtimes(d));
			FunctionAndConstraint fc = funAndConstr.funAndConstr(xNext);
			double fNext = fc.f();
			Matrix bNext = fc.c();

			// Compute merit function at trial point
			double meritFunctionNext = fNext + penalty * safeNorm2(bNext);

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
						d.toColumnArray(), y.toColumnArray(), trustLb.toColumnArray(), trustUb.toColumnArray());
				double t = r.tB();

				// Compute tentative point
				Matrix xSoc = x.plus(S.mtimes(d.plus(y.times(t))));
				FunctionAndConstraint fcSoc = funAndConstr.funAndConstr(xSoc);
				double fSoc = fcSoc.f();
				Matrix bSoc = fcSoc.c();

				// Recompute actual reduction
				double meritFunctionSoc = fSoc + penalty * safeNorm2(bSoc);
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

		return new StatefulResult(x, state, v);
	}
}
