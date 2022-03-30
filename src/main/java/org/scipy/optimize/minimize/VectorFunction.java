package org.scipy.optimize.minimize;

import org.scipy.optimize.minimize.enums.FiniteDifferenceMethod;
import org.scipy.optimize.minimize.interfaces.Function;
import org.scipy.optimize.minimize.interfaces.Hessian;
import org.scipy.optimize.minimize.interfaces.HessianUpdateStrategy;
import org.scipy.optimize.minimize.interfaces.Jacobian;
import org.scipy.optimize.minimize.records.FiniteDifferenceOptions;
import org.ujmp.core.Matrix;

/**
 * Vector function and its derivatives.
 *
 * This class defines a vector function F: R^n->R^m and methods for
 * computing or approximating its first and second derivatives.
 *
 * Notes
 * -----
 * This class implements a memoization logic. There are methods `fun`,
 * `jac`, hess` and corresponding attributes `f`, `J` and `H`. The following
 * things should be considered:
 *
 *     1. Use only public methods `fun`, `jac` and `hess`.
 *     2. After one of the methods is called, the corresponding attribute
 *        will be set. However, a subsequent call with a different argument
 *        of *any* of the methods may overwrite the attribute.
 */
public class VectorFunction {

	public static class VectorFunctionFactory {

		private Function fun;
		private Matrix x0;

		private Jacobian jac;
		private FiniteDifferenceMethod jacFD;
		private boolean hasJac;

		private Hessian hess;
		private FiniteDifferenceMethod hessFD;
		private HessianUpdateStrategy hessStrat;
		private boolean hasHess;

		private double finiteDiffRelStep;
		private FiniteDifferenceBounds finiteDiffBounds;

		private VectorFunctionFactory() {
			hasJac = false;
			hasHess = false;
		}

		/** Set the objective funciton to optimize. */
		public VectorFunctionFactory fun(Function fun) {
			this.fun = fun;
			return this;
		}

		/**
		 * Provides an initial set of variables for evaluating fun.
		 * Array of real elements of size (n,),
		 * where 'n' is the number of independent variables.
		 */
		public VectorFunctionFactory x0(Matrix x0) {
			this.x0 = x0;
			return this;
		}


		/**
		 * Method for computing the gradient vector.
	     * If it is a callable, it should be a function that returns the gradient
	     * vector:
	     *
	     *     ``grad(x, *args) -> array_like, shape (n,)``
	     *
	     * where ``x`` is an array with shape (n,) and ``args`` is a tuple with
	     * the fixed parameters.
	     * Alternatively, the keywords  {'2-point', '3-point', 'cs'} can be used
	     * to select a finite difference scheme for numerical estimation of the
	     * gradient with a relative step size. These finite difference schemes
	     * obey any specified `bounds`.
		 */
		public VectorFunctionFactory jac(Jacobian jac) {
			if (hasJac) {
				throw new RuntimeException("You can only specify either Jacobian or FiniteDifferenceMethod.");
			} else {
				this.jac = jac;
				this.jacFD = null;
				this.hasJac = true;
				return this;
			}
		}

		/**
		 * Method for computing the Jacobian matrix.
		 */
		public VectorFunctionFactory jac(FiniteDifferenceMethod jac) {
			if (hasJac) {
				throw new RuntimeException("You can only specify either Jacobian or FiniteDifferenceMethod.");
			} else {
				this.jac = null;
				this.jacFD = jac;
				this.hasJac = true;
				return this;
			}
		}

		/**
		 * Method for computing the Hessian matrix. If it is callable, it should
	     * return the  Hessian matrix:
	     *
	     *     ``hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)``
	     *
	     * where x is a (n,) ndarray and `args` is a tuple with the fixed
	     * parameters. Alternatively, the keywords {'2-point', '3-point', 'cs'}
	     * select a finite difference scheme for numerical estimation. Or, objects
	     * implementing `HessianUpdateStrategy` interface can be used to
	     * approximate the Hessian.
	     * Whenever the gradient is estimated via finite-differences, the Hessian
	     * cannot be estimated with options {'2-point', '3-point', 'cs'} and needs
	     * to be estimated using one of the quasi-Newton strategies.
		 */
		public VectorFunctionFactory hess(Hessian hess) {
			if (hasHess) {
				throw new RuntimeException("You can only specify either Hessian, FiniteDifferenceMethod or HessianUpdateStrategy.");
			} else {
				this.hess = hess;
				this.hessFD = null;
				this.hessStrat = null;
				this.hasHess = true;
				return this;
			}
		}

		/**
		 * Method for computing the Hessian matrix. If it is callable, it should
	     * return the  Hessian matrix:
	     *
	     *     ``hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)``
	     *
	     * where x is a (n,) ndarray and `args` is a tuple with the fixed
	     * parameters. Alternatively, the keywords {'2-point', '3-point', 'cs'}
	     * select a finite difference scheme for numerical estimation. Or, objects
	     * implementing `HessianUpdateStrategy` interface can be used to
	     * approximate the Hessian.
	     * Whenever the gradient is estimated via finite-differences, the Hessian
	     * cannot be estimated with options {'2-point', '3-point', 'cs'} and needs
	     * to be estimated using one of the quasi-Newton strategies.
		 */
		public VectorFunctionFactory hess(FiniteDifferenceMethod hess) {
			if (hasHess) {
				throw new RuntimeException("You can only specify either Hessian, FiniteDifferenceMethod or HessianUpdateStrategy.");
			} else {
				this.hess = null;
				this.hessFD = hess;
				this.hessStrat = null;
				this.hasHess = true;
				return this;
			}
		}

		/**
		 * Method for computing the Hessian matrix. If it is callable, it should
	     * return the  Hessian matrix:
	     *
	     *     ``hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)``
	     *
	     * where x is a (n,) ndarray and `args` is a tuple with the fixed
	     * parameters. Alternatively, the keywords {'2-point', '3-point', 'cs'}
	     * select a finite difference scheme for numerical estimation. Or, objects
	     * implementing `HessianUpdateStrategy` interface can be used to
	     * approximate the Hessian.
	     * Whenever the gradient is estimated via finite-differences, the Hessian
	     * cannot be estimated with options {'2-point', '3-point', 'cs'} and needs
	     * to be estimated using one of the quasi-Newton strategies.
		 */
		public VectorFunctionFactory hess(HessianUpdateStrategy hess) {
			if (hasHess) {
				throw new RuntimeException("You can only specify either Hessian, FiniteDifferenceMethod or HessianUpdateStrategy.");
			} else {
				this.hess = null;
				this.hessFD = null;
				this.hessStrat = hess;
				this.hasHess = true;
				return this;
			}
		}

		/**
		 * Relative step size to use. The absolute step size is computed as
	     * ``h = finite_diff_rel_step * sign(x0) * max(1, abs(x0))``, possibly
	     * adjusted to fit into the bounds. For ``method='3-point'`` the sign
	     * of `h` is ignored. If None then finite_diff_rel_step is selected
	     * automatically,
		 */
		public VectorFunctionFactory finiteDiffRelStep(double finiteDiffRelStep) {
			this.finiteDiffRelStep = finiteDiffRelStep;
			return this;
		}

		/**
		 * Lower and upper bounds on independent variables. Defaults to no bounds,
	     * (-np.inf, np.inf). Each bound must match the size of `x0` or be a
	     * scalar, in the latter case the bound will be the same for all
	     * variables. Use it to limit the range of function evaluation.
		 */
		public VectorFunctionFactory finiteDiffBounds(FiniteDifferenceBounds finiteDiffBounds) {
			this.finiteDiffBounds = finiteDiffBounds;
			return this;
		}

		public VectorFunction build() {
			// Actually check for nulls to safeguard against calling grad(null) or hess(null).
			if (jac == null && jacFD == null) {
				throw new RuntimeException("jac must be either callable or FiniteDifferenceMethod");
			}

			if (hess == null && hessFD == null && hessStrat == null) {
				throw new RuntimeException("hess must be either callable, FiniteDifferenceMethod or HessianUpdateStrategy");
			}

			if (jacFD != null && hessFD != null) {
				throw new RuntimeException(
						"Whenever the Jacobian is estimated via finite-differences, " +
						"we require the Hessian to be estimated using one of the quasi-Newton strategies (BFGS or SR1).");
			}

			return new VectorFunction(fun, x0,
					jac, jacFD,
					hess, hessFD, hessStrat,
					finiteDiffRelStep, finiteDiffBounds);

		}
	};

	public static final VectorFunctionFactory FACTORY;
	static {
		FACTORY = new VectorFunctionFactory();
	}

	/** current position */
	private Matrix x;

	/** number of parameters */
	private long n;

	private int numFunctionEvals;
	private int numJacobianEvals;
	private int numHessianEvals;

	private Matrix f;
	private boolean updatedF;

	private Matrix J;
	private Matrix JPrev;
	private boolean updatedJ;

	private Matrix H;
	private HessianUpdateStrategy hStrat;
	private boolean updatedH;



	private VectorFunction(Function fun, Matrix x0,
			Jacobian jac, FiniteDifferenceMethod jacFD,
			Hessian hess, FiniteDifferenceMethod hessFD, HessianUpdateStrategy hessStrat,
			double finiteDiffRelStep, FiniteDifferenceBounds finiteDiffBounds) {

		x = Matrix.Factory.copyFromMatrix(x0);
		n = x.getRowCount();

		numFunctionEvals = 0;
		numJacobianEvals = 0;
		numHessianEvals = 0;

		updatedF = false;
		updatedJ = false;
		updatedH = false;

		final FiniteDifferenceOptions options;
		if (jacFD != null) {
			boolean asLinearOperator = false;
			double[] epsilon = null;
			options = new FiniteDifferenceOptions(
					jacFD, finiteDiffRelStep, epsilon, finiteDiffBounds, asLinearOperator);
		} else if (hessFD != null) {
			FiniteDifferenceBounds hessBounds = null;
			boolean asLinearOperator = true;
			double[] epsilon = null;
			options = new FiniteDifferenceOptions(
					hessFD, finiteDiffRelStep, epsilon, hessBounds, asLinearOperator);
		} else {
			options = null;
		}


	}























	public int numFunctionEvals() {
		return numFunctionEvals;
	}

	public int numJacobianEvals() {
		return numJacobianEvals;
	}

	public int numHessianEvals() {
		return numHessianEvals;
	}

	public boolean sparseJacobian() {
		return false;
	}

	public Matrix fun(Matrix x) {
		return null;
	}

	public Matrix jac(Matrix x) {
		return null;
	}

	public Matrix hess(Matrix x, Matrix v) {
		return null;
	}

	public Matrix f() {
		return f;
	}

	public Matrix J() {
		return J;
	}

	public Matrix H() {
		return H;
	}

	public Matrix lastV() {
		return null;
	}
}
