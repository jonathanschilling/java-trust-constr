package org.scipy.optimize.minimize.records;

import java.util.Arrays;

import java.util.Optional;

import org.scipy.optimize.minimize.IdentityVectorFunction;
import org.scipy.optimize.minimize.LinearConstraint;
import org.scipy.optimize.minimize.LinearVectorFunction;
import org.scipy.optimize.minimize.NonlinearConstraint;
import org.scipy.optimize.minimize.NonlinearVectorFunction;
import org.scipy.optimize.minimize.interfaces.VectorFunctionLike;
import org.scipy.optimize.minimize.matrix.Matrix;

/**
 * User-facing constraint prepared for use by the trust-constr orchestrator.
 * Mirrors scipy's {@code PreparedConstraint}
 * ({@code scipy/optimize/_constraints.py:352}): wraps the constraint in an
 * appropriate {@link VectorFunctionLike} carrier (with {@code f} and
 * {@code J} cached at {@code x0}), exposes the bounds and per-row
 * keep-feasible mask, and validates that {@code x0} satisfies any
 * keep-feasible inequality rows.
 *
 * <p>Three constructors handle the three constraint shapes scipy supports:
 * {@link LinearConstraint} (uses {@link LinearVectorFunction}),
 * {@link NonlinearConstraint} (uses {@link NonlinearVectorFunction}),
 * {@link Bounds} (uses {@link IdentityVectorFunction}).
 */
public class PreparedConstraint {

	private final VectorFunctionLike fun;
	private final double[] lb;
	private final double[] ub;
	private final boolean[] keepFeasible;
	private final Matrix x0;
	private final Object source;

	/**
	 * Prepare a {@link LinearConstraint}. {@code f} and {@code J} are cached at
	 * {@code x0} via a {@link LinearVectorFunction}.
	 *
	 * @param lc             linear constraint
	 * @param x0             starting iterate ({@code n x 1})
	 * @param sparseJacobian user preference for Jacobian representation;
	 *                       {@code Optional.empty()} preserves whatever
	 *                       {@code lc.getA()} already is
	 */
	public PreparedConstraint(LinearConstraint lc, Matrix x0,
			Optional<Boolean> sparseJacobian) {
		this.fun = new LinearVectorFunction(lc.getA(), x0, sparseJacobian);
		this.lb = lc.lb();
		this.ub = lc.ub();
		this.keepFeasible = lc.keepFeasible();
		this.x0 = Matrix.Factory.copyFromMatrix(x0);
		this.source = lc;
		validateKeepFeasibleAtX0();
	}

	/**
	 * Prepare a {@link NonlinearConstraint}. {@code f} and {@code J} are cached
	 * at {@code x0} via a {@link NonlinearVectorFunction}.
	 *
	 * @param nc                  nonlinear constraint
	 * @param x0                  starting iterate ({@code n x 1})
	 * @param sparseJacobian      user preference for Jacobian representation
	 * @param finiteDiffBounds    bounds to clamp finite-difference
	 *                            perturbations within (currently unused -- the
	 *                            FD wrapping is done by
	 *                            {@link NonlinearConstraint} itself when the
	 *                            user omits an analytic Jacobian, before this
	 *                            constructor is reached)
	 */
	public PreparedConstraint(NonlinearConstraint nc, Matrix x0,
			Optional<Boolean> sparseJacobian, FiniteDifferenceBounds finiteDiffBounds) {
		this.fun = new NonlinearVectorFunction(
				nc.userFun(), nc.userJac(), nc.userHess(), x0, sparseJacobian);
		this.lb = nc.lb();
		this.ub = nc.ub();
		this.keepFeasible = nc.keepFeasible();
		this.x0 = Matrix.Factory.copyFromMatrix(x0);
		this.source = nc;
		validateKeepFeasibleAtX0();
	}

	/**
	 * Prepare {@link Bounds} (folding them into the constraint set as the
	 * trivial linear constraint {@code lb &le; x &le; ub} with {@code A = I}).
	 * {@code f = x0} and {@code J = I} are cached via an
	 * {@link IdentityVectorFunction}.
	 *
	 * @param bounds         box bounds on the decision variables
	 * @param x0             starting iterate ({@code n x 1})
	 * @param sparseJacobian user preference for Jacobian representation
	 */
	public PreparedConstraint(Bounds bounds, Matrix x0, Optional<Boolean> sparseJacobian) {
		this.fun = new IdentityVectorFunction(x0, sparseJacobian);
		long n = bounds.lb().getRowCount();
		this.lb = new double[(int) n];
		this.ub = new double[(int) n];
		for (int i = 0; i < n; ++i) {
			this.lb[i] = bounds.lb().getAsDouble(i, 0);
			this.ub[i] = bounds.ub().getAsDouble(i, 0);
		}
		// Bounds carries a single scalar keep_feasible flag; broadcast to per-row.
		this.keepFeasible = new boolean[(int) n];
		if (bounds.keepFeasible()) {
			Arrays.fill(this.keepFeasible, true);
		}
		this.x0 = Matrix.Factory.copyFromMatrix(x0);
		this.source = bounds;
		validateKeepFeasibleAtX0();
	}

	/**
	 * Validate that the cached {@code f} satisfies all
	 * {@code keep_feasible & (lb != ub)} rows. Mirrors scipy's check at the
	 * end of {@code PreparedConstraint.__init__}: equality rows ({@code lb ==
	 * ub}) are excluded because they're handled at a different layer.
	 *
	 * @throws IllegalArgumentException if any keep-feasible inequality row is
	 *         violated at {@code x0}
	 */
	private void validateKeepFeasibleAtX0() {
		Matrix f0 = fun.f();
		for (int i = 0; i < keepFeasible.length; ++i) {
			if (!keepFeasible[i]) continue;
			if (lb[i] == ub[i]) continue;
			double v = f0.getAsDouble(i, 0);
			if (v < lb[i] || v > ub[i]) {
				throw new IllegalArgumentException(
						"x0 is infeasible with respect to a keep_feasible "
								+ "inequality row: row " + i + ", lb=" + lb[i]
								+ ", f(x0)[" + i + "]=" + v + ", ub=" + ub[i]);
			}
		}
	}

	/**
	 * Compute the per-row constraint violation at {@code x}. Each row's
	 * violation is {@code max(0, lb - f(x)) + max(0, f(x) - ub)}: the amount
	 * by which the constraint is exceeded on either side. Mirrors scipy's
	 * {@code PreparedConstraint.violation}.
	 *
	 * @param x current iterate ({@code n x 1})
	 * @return per-row violation, length {@link #m()}
	 */
	public double[] violation(Matrix x) {
		Matrix ev = fun.fun(x);
		int len = (int) fun.m();
		double[] excess = new double[len];
		for (int i = 0; i < len; ++i) {
			double v = ev.getAsDouble(i, 0);
			double excessLb = Math.max(lb[i] - v, 0.0);
			double excessUb = Math.max(v - ub[i], 0.0);
			excess[i] = excessLb + excessUb;
		}
		return excess;
	}

	/** @return wrapped {@link VectorFunctionLike} carrier with cached {@code f}, {@code J} */
	public VectorFunctionLike fun() {
		return fun;
	}

	/** @return defensive copy of the lower bounds, length {@link #m()} */
	public double[] lb() {
		return lb.clone();
	}

	/** @return defensive copy of the upper bounds, length {@link #m()} */
	public double[] ub() {
		return ub.clone();
	}

	/** @return defensive copy of the per-row keep-feasible flags, length {@link #m()} */
	public boolean[] keepFeasible() {
		return keepFeasible.clone();
	}

	/** @return number of constraint rows ({@code m}) */
	public long m() {
		return fun.m();
	}

	/** @return number of decision variables ({@code n}) */
	public long n() {
		return fun.n();
	}

	/** @return defensive copy of the iterate {@code x0} the constraint was prepared at */
	public Matrix x0() {
		return Matrix.Factory.copyFromMatrix(x0);
	}

	/**
	 * @return original user-supplied source constraint; one of
	 *         {@link LinearConstraint}, {@link NonlinearConstraint}, or
	 *         {@link Bounds}, depending on which constructor was used
	 */
	public Object source() {
		return source;
	}

	/**
	 * @return bounds repackaged as a {@link Bounds} record
	 *         ({@code lb} and {@code ub} as {@code m x 1} column matrices).
	 *         Mirrors scipy's tuple-shape {@code bounds = (lb, ub)}.
	 */
	public Bounds bounds() {
		double[] lbArr = lb.clone();
		double[] ubArr = ub.clone();
		return new Bounds(Matrix.Factory.linkToArray(lbArr),
				Matrix.Factory.linkToArray(ubArr), false);
	}
}
