package org.scipy.optimize.minimize;

import org.ujmp.core.Matrix;

/**
 * Scalar function and its derivatives.
 * <p>
 * This class defines a scalar function F: R^n->R
 * and methods for computing or approximating its first and second derivatives.
 * <p>
 * This class implements a memoization logic. There are methods `fun`,
 * `grad`, hess` and corresponding attributes `f`, `g` and `H`. The following
 * things should be considered:
 *
 *     1. Use only public methods `fun`, `grad` and `hess`.
 *     2. After one of the methods is called, the corresponding attribute
 *        will be set. However, a subsequent call with a different argument
 *        of *any* of the methods may overwrite the attribute.
 */
public class ScalarFunction {

	public static class ScalarFunctionFactory {

		private Matrix x0;

		private Function fun;

		private Object args;

		private Gradient grad;
		private FiniteDifferenceMethod gradFD;
		private boolean hasGrad;

		private Hessian hess;
		private FiniteDifferenceMethod hessFD;
		private HessianUpdateStrategy hessUpdateStrategy;
		private boolean hasHess;

		private double finiteDiffRelStep;

		private FiniteDifferenceBounds finiteDiffBounds;

		private double[] epsilon;

		private ScalarFunctionFactory() {}

		/**
		 * Provides an initial set of variables for evaluating fun.
		 * Array of real elements of size (n,),
		 * where 'n' is the number of independent variables.
		 */
		public ScalarFunctionFactory x0(Matrix x0) {
			this.x0 = x0;
			return this;
		}

		public ScalarFunctionFactory fun(Function fun) {
			this.fun = fun;
			return this;
		}

		/**
		 * Any additional fixed parameters needed to completely specify the scalar function.
		 */
		public ScalarFunctionFactory args(Object args) {
			this.args = args;
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
		public ScalarFunctionFactory grad(Gradient grad) {
			this.grad = grad;
			this.gradFD = null;
			this.hasGrad = true;
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
		public ScalarFunctionFactory grad(FiniteDifferenceMethod grad) {
			this.grad = null;
			this.gradFD = grad;
			this.hasGrad = true;
			return this;
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
		public ScalarFunctionFactory hess(Hessian hess) {
			this.hess = hess;
			this.hessFD = null;
			this.hasHess = true;
			return this;
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
		public ScalarFunctionFactory hess(FiniteDifferenceMethod hess) {
			this.hess = null;
			this.hessFD = hess;
			this.hasHess = true;
			return this;
		}

		/**
		 * Relative step size to use. The absolute step size is computed as
	     * ``h = finite_diff_rel_step * sign(x0) * max(1, abs(x0))``, possibly
	     * adjusted to fit into the bounds. For ``method='3-point'`` the sign
	     * of `h` is ignored. If None then finite_diff_rel_step is selected
	     * automatically,
		 */
		public ScalarFunctionFactory finiteDiffRelStep(double finiteDiffRelStep) {
			this.finiteDiffRelStep = finiteDiffRelStep;
			return this;
		}

		/**
		 * Lower and upper bounds on independent variables. Defaults to no bounds,
	     * (-np.inf, np.inf). Each bound must match the size of `x0` or be a
	     * scalar, in the latter case the bound will be the same for all
	     * variables. Use it to limit the range of function evaluation.
		 */
		public ScalarFunctionFactory finiteDiffBounds(FiniteDifferenceBounds finiteDiffBounds) {
			this.finiteDiffBounds = finiteDiffBounds;
			return this;
		}

		/**
		 * Absolute step size to use, possibly adjusted to fit into the bounds.
	     * For ``method='3-point'`` the sign of `epsilon` is ignored. By default
	     * relative steps are used, only if ``epsilon is not None`` are absolute
	     * steps used.
		 */
		public ScalarFunctionFactory epsilon(double[] epsilon) {
			this.epsilon = epsilon;
			return this;
		}

		public ScalarFunction build() {


			return new ScalarFunction(fun, x0);
		}
	}

	public static final ScalarFunctionFactory FACTORY;
	static {
		FACTORY = new ScalarFunctionFactory();
	}

	private Function fun;
	private double f;
	private boolean updatedF;
	private Matrix x0;
	private Object args;

	private Gradient grad;
	private FiniteDifferenceMethod gradFD;
	private boolean hasGrad;
	private Matrix g;
	private boolean updatedG;

	private Hessian hess;
	private FiniteDifferenceMethod hessFD;
	private HessianUpdateStrategy hessUpdateStrategy;
	private boolean hasHess;
	private Matrix H;
	private boolean updatedH;

	private double finiteDiffRelStep;
	private FiniteDifferenceBounds finiteDiffBounds;
	private double[] epsilon;


	/** current position */
	private Matrix x;

	/** number of parameters */
	private long n;

	private int numFunctionEvals;
	private int numGradientEvals;
	private int numHessianEvals;

	private Matrix lowestX;
	private double lowestF;


	private ScalarFunction(Function fun, Matrix x0) {
		this.fun = fun;
		this.x0 = x0;
	}



	private void initialize() {
		if (grad == null && gradFD == null) {
			throw new RuntimeException("`grad` must be either callable or one of FiniteDifferenceMethod");
		}

		if (hess == null && hessFD == null && hessUpdateStrategy == null) {
			throw new RuntimeException("hess` must be either callable, HessianUpdateStrategy or one of FiniteDifferenceMethod");
		}

		if (gradFD != null && hessFD != null) {
			throw new RuntimeException(
					"Whenever the gradient is estimated via " +
					"finite-differences, we require the Hessian " +
					"to be estimated using one of the " +
					"quasi-Newton strategies (BFGS or SR1).");
		}

		x = Matrix.Factory.copyFromMatrix(x0);
		n = x.getRowCount();

		numFunctionEvals = 0;
		numGradientEvals = 0;
		numHessianEvals = 0;

		updatedF = false;
		updatedG = false;
		updatedH = false;

		lowestX = null;
		lowestF = Double.POSITIVE_INFINITY;
	}


	public int numFunctionEvals() {
		return numFunctionEvals;
	}

	public int numGradientEvals() {
		return numGradientEvals;
	}

	public int numHessianEvals() {
		return numHessianEvals;
	}

	public double eval(Matrix x) {
		return Double.NaN;
	}

	public Matrix gradient(Matrix x) {
		return null;
	}

	public Matrix hessian(Matrix x) {
		return null;
	}

	public double lastEval() {
		return Double.NaN;
	}

	public Matrix lastGradient() {
		return null;
	}

	public Matrix lastHessian() {
		return null;
	}
}
