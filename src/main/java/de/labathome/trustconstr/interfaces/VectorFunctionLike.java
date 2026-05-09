package de.labathome.trustconstr.interfaces;

import de.labathome.trustconstr.matrix.Matrix;

/**
 * Common surface implemented by every vector-valued function carrier the
 * trust-constr code passes around: the full
 * {@link de.labathome.trustconstr.VectorFunction} (objective wrapper with
 * caching and finite-difference fallbacks),
 * {@link de.labathome.trustconstr.LinearVectorFunction} (constant
 * Jacobian {@code A}), and
 * {@link de.labathome.trustconstr.IdentityVectorFunction} (Jacobian
 * {@code I}).
 *
 * <p>Mirrors the duck-typed contract scipy relies on -- in
 * {@code _constraints.py:PreparedConstraint} the {@code fun} attribute can be
 * any of the three Python classes, and downstream code only calls
 * {@code fun(x)} / {@code jac(x)} / {@code hess(x, v)} or reads the cached
 * {@code f} / {@code J} / {@code m} / {@code n}.
 */
public interface VectorFunctionLike {

	/**
	 * Evaluate the function at {@code x}, updating any internal cache.
	 *
	 * @param x current iterate ({@code n x 1})
	 * @return function value ({@code m x 1})
	 */
	Matrix fun(Matrix x);

	/**
	 * Evaluate the Jacobian at {@code x}, updating any internal cache.
	 *
	 * @param x current iterate ({@code n x 1})
	 * @return Jacobian ({@code m x n})
	 */
	Matrix jac(Matrix x);

	/**
	 * Evaluate the constraint Hessian-of-Lagrangian
	 * {@code Sum_i v[i] * H_{f_i}(x)} at {@code x} for the given multiplier
	 * vector {@code v}.
	 *
	 * @param x current iterate ({@code n x 1})
	 * @param v multipliers ({@code m x 1})
	 * @return summed Hessian ({@code n x n})
	 */
	Matrix hess(Matrix x, Matrix v);

	/** @return most recently computed {@code f(x)} ({@code m x 1}), or {@code null} if uncached */
	Matrix f();

	/** @return most recently computed Jacobian ({@code m x n}), or {@code null} if uncached */
	Matrix J();

	/** @return number of output components ({@code m} in {@code f : R^n -> R^m}) */
	long m();

	/** @return number of input components ({@code n} in {@code f : R^n -> R^m}) */
	long n();

	/** @return total number of function evaluations performed so far */
	default int numFunctionEvals() {
		return 0;
	}

	/** @return total number of Jacobian evaluations performed so far */
	default int numJacobianEvals() {
		return 0;
	}

	/** @return total number of Hessian evaluations performed so far */
	default int numHessianEvals() {
		return 0;
	}

	/**
	 * @return most recently set Lagrange multipliers (from
	 *         {@link #hess(Matrix, Matrix)}); shape {@code m x 1}, or
	 *         {@code null} if none ever set
	 */
	Matrix v();
}
