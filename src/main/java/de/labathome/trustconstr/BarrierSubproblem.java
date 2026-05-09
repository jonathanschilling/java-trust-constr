package de.labathome.trustconstr;

import java.util.function.BiFunction;
import java.util.function.ToDoubleBiFunction;

import de.labathome.trustconstr.interfaces.Constraint;
import de.labathome.trustconstr.interfaces.GlobalStoppingCriteria;
import de.labathome.trustconstr.interfaces.Jacobian;
import de.labathome.trustconstr.interfaces.LagrangeHessian;
import de.labathome.trustconstr.interfaces.LinearOperator;
import de.labathome.trustconstr.records.CGInfo;
import de.labathome.trustconstr.records.FunctionAndConstraint;
import de.labathome.trustconstr.records.GradientAndJacobian;
import de.labathome.trustconstr.records.State;
import de.labathome.trustconstr.sparse.CSRMatrix;
import de.labathome.trustconstr.sparse.SparseAssembly;

import de.labathome.trustconstr.matrix.Matrix;
import de.labathome.trustconstr.matrix.SparseMatrix;

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

	/**
	 * Construct the barrier subproblem at a starting iterate.
	 *
	 * @param x0     starting point in the variable space ({@code nVars x 1})
	 * @param s0     starting slacks ({@code nIneq x 1}); must be strictly positive
	 * @param fun    objective {@code (x, args) -> f(x)}
	 * @param grad   gradient {@code (x, args) -> gradf(x)}
	 * @param lagrHess Hessian-of-Lagrangian {@code (x, v) -> grad^2L(x, v)}
	 * @param nVars  number of decision variables
	 * @param nIneq  number of canonical inequality rows
	 * @param nEq    number of canonical equality rows
	 * @param constr combined-constraint evaluator
	 * @param jac    combined-constraint Jacobian evaluator
	 * @param barrierParameter initial log-barrier coefficient
	 * @param tolerance        initial inner-loop tolerance
	 * @param enforceFeasibility per-canonical-ineq strict-feasibility flag; may be {@code null}
	 * @param globalStopCriteria outer-loop termination predicate
	 * @param xtol   trust-region radius termination threshold
	 * @param fun0   {@code fun(x0)}
	 * @param grad0  {@code grad(x0)}
	 * @param constrIneq0 {@code constr_ineq(x0)} ({@code nIneq x 1})
	 * @param jacIneq0    inequality Jacobian at {@code x0} ({@code nIneq x n})
	 * @param constrEq0   {@code constr_eq(x0)} ({@code nEq x 1})
	 * @param jacEq0      equality Jacobian at {@code x0} ({@code nEq x n})
	 */
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

	/**
	 * Refresh the barrier coefficient and inner-loop tolerance for the next
	 * outer iteration.
	 *
	 * @param barrierParameter new log-barrier coefficient
	 * @param tolerance new inner-loop tolerance
	 */
	public void update(double barrierParameter, double tolerance) {
		this.barrierParameter = barrierParameter;
		this.tolerance = tolerance;
	}

	/**
	 * Slice off the {@code nIneq x 1} slack block from an augmented iterate
	 * {@code z = [x; s]}.
	 *
	 * @param z augmented iterate
	 * @return slack subvector {@code s}
	 */
	public Matrix getSlack(Matrix z) {
		return z.subMatrix(nVars, 0, nVars+nIneq-1, 0);
	}

	/**
	 * Slice off the {@code nVars x 1} variable block from an augmented
	 * iterate {@code z = [x; s]}.
	 *
	 * @param z augmented iterate
	 * @return variable subvector {@code x}
	 */
	public Matrix getVariables(Matrix z) {
		return z.subMatrix(0, 0, nVars-1, 0);
	}

	/**
	 * Evaluate the barrier function and the augmented constraints at a given
	 * augmented iterate {@code z = [x; s]}:
	 * <pre>
	 *   function(z)    = fun(x) - barrier_parameter * sum(log(s))
	 *   constraints(z) = [   constr_eq(x)     ]
	 *                    [ constr_ineq(x) + s ]
	 * </pre>
	 *
	 * @param z augmented iterate {@code [x; s]} ({@code (nVars + nIneq) x 1})
	 * @return barrier objective value and augmented constraint vector
	 */
	public FunctionAndConstraint funAndConstr(Matrix z) {

		// Get variables and slack variables
		Matrix x = getVariables(z);
		Matrix s = getSlack(z);

		// Compute function and constraints. The `args` slot is intentionally
		// null -- BarrierSubproblem doesn't track scipy's `args`; the
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
	 * Returns the diagonal scaling matrix {@code diag([1...1, s])} that
	 * decouples the variable rows from the slack rows in the trust-region
	 * step. Identity on the variable rows; the current slack values on the
	 * slack rows.
	 *
	 * @param z augmented iterate {@code [x; s]}
	 * @return {@code (nVars + nIneq) x (nVars + nIneq)} diagonal scaling
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
	 * Returns the scaled gradient and Jacobian for the augmented system:
	 * <pre>
	 *   gradient = [             grad(x)             ]
	 *              [ -barrier_parameter*ones(n_ineq) ]
	 *   jacobian = [  jac_eq(x)  0 ]
	 *              [ jac_ineq(x) S ]
	 * </pre>
	 * Both rescaled by the diagonal returned from {@link #getScaling(Matrix)}.
	 *
	 * @param z augmented iterate {@code [x; s]}
	 * @return barrier-augmented gradient and Jacobian
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
	 * Lagrangian Hessian with respect to {@code x} (the {@code Hx} block).
	 *
	 * @param z augmented iterate {@code [x; s]}
	 * @param v full Lagrange-multiplier vector (equality + inequality)
	 * @return {@code Hx} as a {@link LinearOperator}
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
	 * Scaled Lagrangian Hessian with respect to the slacks {@code s}, i.e.
	 * the {@code S Hs S} block. Uses the primal-dual formulation for entries
	 * with positive {@code v_ineq} and the primal formulation otherwise.
	 *
	 * @param z augmented iterate {@code [x; s]}
	 * @param v full Lagrange-multiplier vector (equality then inequality)
	 * @return {@code nIneq x 1} diagonal of the {@code S Hs S} block
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
		Matrix subV = v.subMatrix(nEq, 0, nEq+nIneq-1, 0);
		Matrix primalDual = subV.times(s);

		// Uses the primal-dual formulation for
        // positives values of v_ineq, and primal
        // formulation for the remaining ones.
		int retRows = (int) subV.getRowCount();
		int retCols = (int) subV.getColumnCount();
		Matrix ret = Matrix.Factory.zeros(retRows, retCols);
		for (int i = 0; i < retRows; ++i) {
			for (int j = 0; j < retCols; ++j) {
				double vVal = subV.getAsDouble(i, j);
				if (vVal > 0.0) {
					ret.setAsDouble(primalDual.getAsDouble(i, j), i, j);
				} else {
					ret.setAsDouble(primal, i, j);
				}
			}
		}

		return ret;
	}

	/**
	 * Full augmented Lagrangian Hessian, block-diagonal:
	 * <pre>
	 *   [ Hx    0     ]
	 *   [  0  S Hs S  ]
	 * </pre>
	 *
	 * @param z augmented iterate {@code [x; s]}
	 * @param v full Lagrange-multiplier vector (equality + inequality)
	 * @return {@link LinearOperator} applying the augmented Hessian
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
	 * Stopping criterion for the barrier subproblem; mirrors formula (2.3)
	 * from Byrd-Hribar-Nocedal (1999), p.879.
	 *
	 * @param state outer-loop state
	 * @param z augmented iterate {@code [x; s]}
	 * @param lastIterationFailed whether the previous step was rejected
	 * @param optimality KKT optimality measure at {@code z}
	 * @param constrViolation infinity-norm of the constraint residual
	 * @param trustRadius current trust-region radius
	 * @param penalty current merit-function penalty
	 * @param cgInfo info from the projected-CG inner solve
	 * @return {@code true} if the inner barrier loop should terminate
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
	 * {@link SparseAssembly#assembleJacobianWithSlacks} -- this is the optimised
	 * counterpart of the generic {@code block_array} call that scipy comments
	 * about. The output is wrapped back into a {@link Matrix} so callers see
	 * the same type they always have.
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
