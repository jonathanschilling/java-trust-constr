package org.scipy.optimize.minimize;

import java.util.function.Function;

import org.scipy.optimize.minimize.interfaces.LinearOperator;
import org.scipy.optimize.minimize.matrix.Matrix;

/**
 * Lagrangian Hessian {@code H_obj(x) + Sum_i v_i * H_{c_i}(x)} as a
 * matrix-free {@link LinearOperator}-builder. Counterpart to scipy's
 * {@code LagrangianHessian} class
 * ({@code _trustregion_constr/minimize_trustregion_constr.py:35}).
 *
 * <p>Construct with the objective Hessian callable and a constraint
 * Hessian-of-Lagrangian callable; invoke {@link #apply(Matrix, double[], double[])}
 * to obtain a {@link LinearOperator} that evaluates {@code (H_obj + H_constr) * p}
 * for a given {@code p}.
 */
public final class LagrangianHessian {

	/**
	 * Constraint Hessian-of-Lagrangian:
	 * {@code Sum_i v_eq[i] * H_{eq,i}(x) + Sum_j v_ineq[j] * H_{ineq,j}(x)}.
	 */
	@FunctionalInterface
	public interface ConstraintHessian {
		/**
		 * @param x     current iterate ({@code n x 1})
		 * @param vEq   equality multipliers (may be {@code null} or empty)
		 * @param vIneq inequality multipliers (may be {@code null} or empty)
		 * @return summed constraint Hessian {@code n x n}, or {@code null}
		 *         when no constraint contributes (e.g. all-linear constraint set)
		 */
		Matrix apply(Matrix x, double[] vEq, double[] vIneq);
	}

	private final int n;
	private final Function<Matrix, Matrix> objectiveHess;
	private final ConstraintHessian constraintsHess;

	/**
	 * Build a Lagrangian Hessian for an {@code n}-variable problem.
	 *
	 * @param n               number of decision variables
	 * @param objectiveHess   objective Hessian callable {@code x -> H_obj(x)}
	 * @param constraintsHess constraint Hessian-of-Lagrangian callable
	 *                        {@code (x, vEq, vIneq) -> Sum_i v_i H_{c_i}(x)};
	 *                        may return {@code null} when no constraint
	 *                        contributes (the apply() result is then just
	 *                        the objective Hessian)
	 */
	public LagrangianHessian(int n, Function<Matrix, Matrix> objectiveHess,
			ConstraintHessian constraintsHess) {
		this.n = n;
		this.objectiveHess = objectiveHess;
		this.constraintsHess = constraintsHess;
	}

	/**
	 * @param x     current iterate ({@code n x 1})
	 * @param vEq   equality multipliers (may be {@code null} or empty)
	 * @param vIneq inequality multipliers (may be {@code null} or empty)
	 * @return matrix-free linear operator that applies
	 *         {@code (H_obj(x) + H_constr(x, vEq, vIneq)) * p} to a given
	 *         input vector {@code p}
	 */
	public LinearOperator apply(Matrix x, double[] vEq, double[] vIneq) {
		Matrix Hobj = objectiveHess.apply(x);
		Matrix Hconstr = (constraintsHess != null)
				? constraintsHess.apply(x, vEq, vIneq)
				: null;
		final Matrix Hfinal = (Hconstr != null) ? Hobj.plus(Hconstr) : Hobj;
		return p -> Hfinal.mtimes(p);
	}

	/** @return number of decision variables this Hessian is sized for */
	public int n() {
		return n;
	}
}
