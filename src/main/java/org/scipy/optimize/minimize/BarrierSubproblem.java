package org.scipy.optimize.minimize;

import org.ujmp.core.Matrix;
import org.ujmp.core.SparseMatrix;
import org.ujmp.core.calculation.Calculation.Ret;

/**
 * Barrier optimization problem:
 *
 * <pre>
 * minimize fun(x) - barrier_parameter*sum(log(s))
 * subject to: constr_eq(x)     = 0
 *           constr_ineq(x) + s = 0
 * </pre>
 */
public class BarrierSubproblem {

	long nVars;
	Matrix x0;
	Matrix s0;
	Function fun;
	Gradient grad;
	LagrangeHessian lagrHess;
	Constraint constr;
	Jacobian jac;
	double barrierParameter;
	double tolerance;
	long nEq;
	long nIneq;
	boolean[] enforceFeasibility;
	GlobalStoppingCriteria globalStopCriteria;
	double xtol;
	double fun0;
	Matrix grad0;
	Matrix constr0;
	Matrix jac0;
	boolean terminate;

	public BarrierSubproblem(Matrix x0, Matrix s0,
			Function fun, Gradient grad, LagrangeHessian lagrHess,
			long nVars, long nIneq, long nEq,
			Constraint constr, Jacobian jac,
			double barrierParameter, double tolerance,
			boolean[] enforceFeasibility, GlobalStoppingCriteria globalStopCriteria,
			double xtol, double fun0, Matrix grad0,
			Matrix constrIneq0, Matrix jacIneq0,
			Matrix constrEq0, Matrix jacEq0) {

		// Store parameters
		this.nVars = nVars;
		this.x0 = x0;
		this.s0 = s0;
		this.fun = fun;
		this.grad = grad;
		this.lagrHess = lagrHess;
		this.constr = constr;
		this.jac = jac;
		this.barrierParameter = barrierParameter;
		this.tolerance = tolerance;
		this.nEq = nEq;
		this.nIneq = nIneq;
		this.enforceFeasibility = enforceFeasibility;
		this.globalStopCriteria = globalStopCriteria;
		this.xtol = xtol;

		this.fun0 = computeFunction(fun0, constrIneq0, s0);
		this.grad0 = computeGradient(grad0);
		this.constr0 = computeConstraint(constrIneq0, constrEq0, s0);
		this.jac0 = computeJacobian(jacEq0, jacIneq0, s0);

		this.terminate = false;
	}

	public void update(double barrierParameter, double tolerance) {
		this.barrierParameter = barrierParameter;
		this.tolerance = tolerance;
	}

	public Matrix getSlack(Matrix z) {
		return z.subMatrix(Ret.NEW, nVars, 0, nVars+nIneq-1, 0);
	}

	public Matrix getVariables(Matrix z) {
		return z.subMatrix(Ret.NEW, 0, 0, nVars-1, 0);
	}

	/**
	 * Returns barrier function and constraints at given point.
	 *
	 * For z = [x, s], returns barrier function:
	 * <pre>
	 *     function(z) = fun(x) - barrier_parameter*sum(log(s))
	 * </pre>
	 * and barrier constraints:
	 * <pre>
	 *     constraints(z) = [   constr_eq(x)     ]
	 *                      [ constr_ineq(x) + s ]
	 * </pre>
	 *
	 * @param z
	 * @return
	 */
	public FunctionAndConstraint funAndConstr(Matrix z) {

		// Get variables and slack variables
		Matrix x = getVariables(z);
		Matrix s = getSlack(z);

		// Compute function and constraints
		double f = fun.fun(x);
		Matrix cEq = constr.constrEq(x);
		Matrix cIneq = constr.constrIneq(x);

		// Return objective function and constraints
		FunctionAndConstraint fc = new FunctionAndConstraint();
		fc.f = computeFunction(f, cIneq, s);
		fc.c = computeConstraint(cIneq, cEq, s);
		return fc;
	}

	/**
	 * Returns scaling vector.
	 * Given by:
	 *     scaling = [ones(n_vars), s]
	 * @param z
	 * @return
	 */
	public Matrix getScaling(Matrix z) {
		Matrix s = getSlack(z);
		Matrix diagElements = SparseMatrix.Factory.zeros(nVars + s.getRowCount(), nVars + s.getColumnCount());
		for (int i=0; i<nVars; ++i) {
			diagElements.setAsDouble(1.0, i, i);
		}
		for (long[] pos: s.allCoordinates()) {
			diagElements.setAsDouble(s.getAsDouble(pos), pos[0], pos[0]);
		}

		return diagElements;
	}

	/**
	 * Returns scaled gradient.
	 *
	 * Return scaled gradient:
	 * <pre>
     *      gradient = [             grad(x)             ]
     *                 [ -barrier_parameter*ones(n_ineq) ]
     * </pre>
     * and scaled Jacobian matrix:
     * <pre>
     *      jacobian = [  jac_eq(x)  0  ]
     *                 [ jac_ineq(x) S  ]
     * </pre>
     * Both of them scaled by the previously defined scaling factor.
     *
	 * @param z
	 * @return
	 */
	public GradientAndJacobian gradAndJac(Matrix z) {

		// Get variables and slack variables
		Matrix x = getVariables(z);
		Matrix s = getSlack(z);

		// Compute first derivatives
		Matrix g = grad.grad(x);
		Matrix jEq = jac.jacEq(x);
		Matrix jIneq = jac.jacIneq(x);

		// Return gradient and Jacobian
		GradientAndJacobian gj = new GradientAndJacobian();
		gj.grad = computeGradient(g);
		gj.jac = computeJacobian(jEq, jIneq, s);
		return gj;
	}

	/**
	 * Returns Lagrangian Hessian (in relation to `x`) -> Hx
	 *
	 * @param z
	 * @param v
	 * @return
	 */
	public LinearOperator lagrHessX(Matrix z, Matrix v) {
		Matrix x = getVariables(z);

//		  # Get lagrange multipliers relatated to nonlinear equality constraints
//        v_eq = v[:self.n_eq]
//
//        # Get lagrange multipliers relatated to nonlinear ineq. constraints
//        v_ineq = v[self.n_eq:self.n_eq+self.n_ineq]

		// --> lagrHess wants all Lagrange multipliers in one vector already...
		return lagrHess.lagrHess(x, v);
	}

	/**
	 * Returns scaled Lagrangian Hessian (in relation to`s`) -> S Hs S
	 *
	 * @param z
	 * @param v
	 * @return
	 */
	public Matrix lagrHessS(Matrix z, Matrix v) {

		Matrix s = getSlack(z);

		// Using the primal formulation:
        //     S Hs S = diag(s)*diag(barrier_parameter/s**2)*diag(s).
        // Reference [1] p. 882, formula (3.1)
		double primal = barrierParameter;

		// Using the primal-dual formulation
        //     S Hs S = diag(s)*diag(v/s)*diag(s)
        // Reference [1] p. 883, formula (3.11)
		Matrix subV = v.subMatrix(Ret.NEW, nEq, 0, nEq+nIneq-1, 0);
		Matrix primalDual = subV.times(s);

		// Uses the primal-dual formulation for
        // positives values of v_ineq, and primal
        // formulation for the remaining ones.
		Matrix ret = Matrix.Factory.zeros(subV.getRowCount(), subV.getColumnCount());
		for (long[] pos: ret.allCoordinates()) {
			double vVal = subV.getAsDouble(pos);
			if (vVal > 0.0) {
				ret.setAsDouble(primalDual.getAsDouble(pos), pos);
			} else {
				ret.setAsDouble(primal, pos);
			}
		}

		return ret;
	}

	/**
	 * Returns scaled Lagrangian Hessian
	 *
	 * @param z
	 * @param v
	 * @return
	 */
	public LinearOperator lagrangianHessian(Matrix z, Matrix v) {

		// Compute Hessian in relation to x and s
		LinearOperator Hx = lagrHessX(z, v);
		Matrix S_Hs_S;
		if (nIneq > 0) {
			S_Hs_S = lagrHessS(z, v);
		} else {
			S_Hs_S = null;
		}

		// The scaled Lagragian Hessian is:
        //     [ Hx    0    ]
        //     [ 0   S Hs S ]
		return new LinearOperator() {

			@Override
			public Matrix apply(Matrix vec) {
				Matrix vecX = getVariables(vec);
				Matrix vecS = getSlack(vec);
				if (nIneq > 0) {
					return Matrix.Factory.vertCat(Hx.apply(vecX), S_Hs_S.times(vecS));
				} else {
					return Hx.apply(vecX);
				}
			}
		};
	}

	/**
	 * Stop criteria to the barrier problem.
	 * The criteria here proposed is similar to formula (2.3) from [1], p.879.
	 *
	 * @param state
	 * @param z
	 * @param lastIterationFailed
	 * @param optimality
	 * @param constrViolation
	 * @param trustRadius
	 * @param penalty
	 * @param cgInfo
	 * @return
	 */
	public boolean stoppingCriteria(State state, Matrix z, boolean lastIterationFailed,
			double optimality, double constrViolation,
			double trustRadius, double penalty, CGInfo cgInfo) {

		Matrix x = getVariables(z);

		if (globalStopCriteria.shouldStop(state, x, lastIterationFailed,
				optimality, constrViolation, trustRadius, penalty, cgInfo,
				barrierParameter, tolerance)) {
			terminate = true;
			return true;
		} else {
			boolean gCond = (optimality < tolerance && constrViolation < tolerance);
			boolean xCond = trustRadius < xtol;
			return (gCond || xCond);
		}
	}

	/**
	 * Use technique from Nocedal and Wright book, ref [3]_, p.576,
	 * to guarantee constraints from `enforce_feasibility`
	 * stay feasible along iterations.
	 *
	 * @param f
	 * @param cIneq
	 * @param s
	 * @return
	 */
	private double computeFunction(double f, Matrix cIneq, Matrix s) {

		double sumLogS = 0.0;
		for (int i=0; i<enforceFeasibility.length; ++i) {
			if (enforceFeasibility[i]) {
				s.setAsDouble(-cIneq.getAsDouble(i, 0), i, 0);
			}

			double sI = s.getAsDouble(i, 0);
			if (sI > 0.0) {
				sumLogS += Math.log(sI);
			} else {
				sumLogS += Double.NEGATIVE_INFINITY;
			}
		}

		return f - barrierParameter * sumLogS;
	}

	private Matrix computeGradient(Matrix g) {
		return Matrix.Factory.vertCat(g, Matrix.Factory.ones(nIneq, 1).times(-barrierParameter));
	}

	/**
	 * Compute barrier constraint
	 *
	 * @param cIneq
	 * @param cEq
	 * @param s
	 * @return
	 */
	private Matrix computeConstraint(Matrix cIneq, Matrix cEq, Matrix s) {
		return Matrix.Factory.vertCat(cEq, cIneq.plus(s));
	}

	private Matrix computeJacobian(Matrix jEq, Matrix jIneq, Matrix s) {
		if (nIneq == 0) {
			return jEq;
		} else {
//			if (jEq.isSparse() || jIneq.isSparse()) {
//				 // It is expected that J_eq and J_ineq
//	             // are already `csr_matrix` because of
//	             // the way ``BoxConstraint``, ``NonlinearConstraint``
//	             // and ``LinearConstraint`` are defined.
//				return assembleSparseJacobian(jEq, jIneq, s);
//			} else {
				Matrix S = SparseMatrix.Factory.zeros(s.getRowCount(), s.getRowCount());
				for (long[] pos: s.allCoordinates()) {
					S.setAsDouble(s.getAsDouble(pos), pos[0], pos[0]);
				}
				Matrix zeros = Matrix.Factory.zeros(nEq, nIneq);

				return Matrix.Factory.vertCat(
						Matrix.Factory.horCat(jEq, zeros),
						Matrix.Factory.horCat(jIneq, S));
//			}
		}
	}

//	/**
//	 * Assemble sparse Jacobian given its components.
//     *
//     * Given {@code J_eq}, {@code J_ineq} and {@code s} returns:
//     * <pre>
//     *     jacobian = [ J_eq,     0     ]
//     *                [ J_ineq, diag(s) ]
//     * </pre>
//     * It is equivalent to:
//     * <pre>
//     *     sps.bmat([[ J_eq,   None    ],
//     *               [ J_ineq, diag(s) ]], "csr")
//     * </pre>
//     * but significantly more efficient for this given structure.
//     *
//	 * @param jEq
//	 * @param jIneq
//	 * @param s
//	 * @return
//	 */
//	private Matrix assembleSparseJacobian(Matrix jEq, Matrix jIneq, Matrix s) {
//		// special case optimization from SciPy not applicable here
//		return null;
//	}
}
