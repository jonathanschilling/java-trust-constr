package org.scipy.optimize.minimize;

import java.util.function.BiFunction;
import java.util.function.ToDoubleBiFunction;

import org.scipy.optimize.minimize.interfaces.Constraint;
import org.scipy.optimize.minimize.interfaces.GlobalStoppingCriteria;
import org.scipy.optimize.minimize.interfaces.Jacobian;
import org.scipy.optimize.minimize.interfaces.LagrangeHessian;
import org.scipy.optimize.minimize.interfaces.LinearOperator;
import org.scipy.optimize.minimize.records.CGInfo;
import org.scipy.optimize.minimize.records.FunctionAndConstraint;
import org.scipy.optimize.minimize.records.GradientAndJacobian;
import org.scipy.optimize.minimize.records.State;
import org.scipy.optimize.minimize.sparse.CSRMatrix;
import org.scipy.optimize.minimize.sparse.SparseAssembly;

import org.scipy.optimize.minimize.matrix.Matrix;
import org.scipy.optimize.minimize.matrix.SparseMatrix;
import org.scipy.optimize.minimize.matrix.Ret;

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
	ToDoubleBiFunction<Matrix, Object> fun;
	BiFunction<Matrix, Object, Matrix> grad;
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
			ToDoubleBiFunction<Matrix, Object> fun, BiFunction<Matrix, Object, Matrix> grad, LagrangeHessian lagrHess,
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

		// Compute function and constraints. The `args` slot is intentionally
		// null — BarrierSubproblem doesn't track scipy's `args`; the
		// orchestrator (MinimizeTrustConstr) bakes them into `fun` via a
		// closure before reaching this code path, so the second
		// argument here is unused by the wrapper.
		double f = fun.applyAsDouble(x, null);
		Matrix cEq = constr.constrEq(x);
		Matrix cIneq = constr.constrIneq(x);

		// Return objective function and constraints
		FunctionAndConstraint fc = new FunctionAndConstraint(
				computeFunction(f, cIneq, s),
				computeConstraint(cIneq, cEq, s));
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
		// Square diagonal of size (n_vars + n_ineq): identity on the variable rows,
		// the current slack values on the slack rows.
		long total = nVars + s.getRowCount();
		Matrix diagElements = SparseMatrix.Factory.zeros(total, total);
		for (int i = 0; i < nVars; ++i) {
			diagElements.setAsDouble(1.0, i, i);
		}
		for (int i = 0; i < s.getRowCount(); ++i) {
			diagElements.setAsDouble(s.getAsDouble(i, 0), nVars + i, nVars + i);
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

		// Compute first derivatives. `args` slot is null for the same
		// closure-baked reason as in funAndConstr above.
		Matrix g = grad.apply(x, null);
		Matrix jEq = jac.jacEq(x);
		Matrix jIneq = jac.jacIneq(x);

		// Return gradient and Jacobian
		GradientAndJacobian gj = new GradientAndJacobian(
				computeGradient(g),
				computeJacobian(jEq, jIneq, s));
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

	/**
	 * Assemble the augmented Jacobian {@code [[J_eq, 0], [J_ineq, diag(s)]]} for
	 * the barrier subproblem ({@code tr_interior_point.py:_assemble_sparse_jacobian}).
	 *
	 * The block is assembled directly in CSR via
	 * {@link SparseAssembly#assembleJacobianWithSlacks} — this is the optimised
	 * counterpart of the generic {@code block_array} call that scipy comments
	 * about. The output is converted back to a UJMP {@link Matrix} so existing
	 * callers see the same type they always have.
	 */
	private Matrix computeJacobian(Matrix jEq, Matrix jIneq, Matrix s) {
		if (nIneq == 0) {
			return jEq;
		}
		CSRMatrix jEqCsr = CSRMatrix.fromMatrix(jEq);
		CSRMatrix jIneqCsr = CSRMatrix.fromMatrix(jIneq);
		double[] sArr = s.toColumnArray();
		CSRMatrix combined = SparseAssembly.assembleJacobianWithSlacks(jEqCsr, jIneqCsr, sArr);
		return Matrix.Factory.linkToArray(combined.toDense());
	}
}
