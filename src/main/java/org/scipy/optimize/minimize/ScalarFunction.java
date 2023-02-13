package org.scipy.optimize.minimize;

import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.ToDoubleBiFunction;
import java.util.function.ToDoubleFunction;

import org.scipy.optimize.minimize.enums.FiniteDifferenceMethod;
import org.scipy.optimize.minimize.enums.HessianApproximationType;
import org.scipy.optimize.minimize.interfaces.HessianUpdateStrategy;
import org.scipy.optimize.minimize.records.FiniteDifferenceBounds;
import org.scipy.optimize.minimize.records.FiniteDifferenceOptions;
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

		private ToDoubleBiFunction<Matrix, Object> fun;
		private Matrix x0;
		private Object args;

		private BiFunction<Matrix, Object, Matrix> grad;
		private FiniteDifferenceMethod gradFD;
		private boolean hasGrad;

		private BiFunction<Matrix, Object, Matrix> hess;
		private FiniteDifferenceMethod hessFD;
		private HessianUpdateStrategy hessStrat;
		private boolean hasHess;

		private Matrix finiteDiffRelStep;
		private FiniteDifferenceBounds finiteDiffBounds;
		private Matrix epsilon;

		private ScalarFunctionFactory() {
			hasGrad = false;
			hasHess = false;
		}

		/** Set the objective function to optimize. */
		public ScalarFunctionFactory fun(ToDoubleBiFunction<Matrix, Object> fun) {
			this.fun = fun;
			return this;
		}

		/**
		 * Provides an initial set of variables for evaluating fun.
		 * Array of real elements of size (n,),
		 * where 'n' is the number of independent variables.
		 */
		public ScalarFunctionFactory x0(Matrix x0) {
			this.x0 = x0;
			return this;
		}

		/** Any additional fixed parameters needed to completely specify the scalar function. */
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
		public ScalarFunctionFactory grad(BiFunction<Matrix, Object, Matrix> grad) {
			if (hasGrad) {
				throw new RuntimeException("You can only specify either Gradient or FiniteDifferenceMethod.");
			} else {
				this.grad = grad;
				this.gradFD = null;
				this.hasGrad = true;
				return this;
			}
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
			if (hasGrad) {
				throw new RuntimeException("You can only specify either Gradient or FiniteDifferenceMethod.");
			} else {
				this.grad = null;
				this.gradFD = grad;
				this.hasGrad = true;
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
		public ScalarFunctionFactory hess(BiFunction<Matrix, Object, Matrix> hess) {
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
		public ScalarFunctionFactory hess(FiniteDifferenceMethod hess) {
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
		public ScalarFunctionFactory hess(HessianUpdateStrategy hess) {
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
		public ScalarFunctionFactory finiteDiffRelStep(Matrix finiteDiffRelStep) {
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
		public ScalarFunctionFactory epsilon(Matrix epsilon) {
			this.epsilon = epsilon;
			return this;
		}

		public ScalarFunction build() {

			// Actually check for nulls to safeguard against calling grad(null) or hess(null).
			if (grad == null && gradFD == null) {
				throw new RuntimeException("grad must be either callable or FiniteDifferenceMethod");
			}

			if (hess == null && hessFD == null && hessStrat == null) {
				throw new RuntimeException("hess must be either callable, FiniteDifferenceMethod or HessianUpdateStrategy");
			}

			if (gradFD != null && hessFD != null) {
				throw new RuntimeException(
						"Whenever the gradient is estimated via finite-differences, " +
						"we require the Hessian to be estimated using one of the quasi-Newton strategies (BFGS or SR1).");
			}

			return new ScalarFunction(fun, x0, args,
					grad, gradFD,
					hess, hessFD, hessStrat,
					finiteDiffRelStep, finiteDiffBounds, epsilon);
		}
	}

	public static final ScalarFunctionFactory FACTORY;
	static {
		FACTORY = new ScalarFunctionFactory();
	}

	/** current position */
	private Matrix x;
	private Matrix xPrev;

	/** number of parameters */
	private long n;

	private int numFunctionEvals;
	private int numGradientEvals;
	private int numHessianEvals;

	private double f;
	private boolean updatedF;

	private Matrix g;
	private Matrix gPrev;
	private boolean updatedG;

	private Matrix H;
	private HessianUpdateStrategy hStrat;
	private boolean updatedH;

	private double lowestF;
	private Matrix lowestX;

	private Runnable updateFunImpl;
	private Runnable updateGradImpl;
	private Runnable updateHessImpl;
	private Consumer<Matrix> updateXImpl;

	private ScalarFunction(ToDoubleBiFunction<Matrix, Object> fun, Matrix x0, Object args,
			BiFunction<Matrix, Object, Matrix> grad, FiniteDifferenceMethod gradFD,
			BiFunction<Matrix, Object, Matrix> hess, FiniteDifferenceMethod hessFD, HessianUpdateStrategy hessStrat,
			Matrix finiteDiffRelStep, FiniteDifferenceBounds finiteDiffBounds, Matrix epsilon) {

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

		final FiniteDifferenceOptions options;
		if (gradFD != null) {
			options = FiniteDifferenceOptions.FACTORY
					.method(gradFD)
					.relStep(finiteDiffRelStep)
					.absStep(epsilon)
					.bounds(finiteDiffBounds)
					.build();
		} else if (hessFD != null) {
			options = FiniteDifferenceOptions.FACTORY
					.method(hessFD)
					.relStep(finiteDiffRelStep)
					.absStep(epsilon)
					.asLinearOperator(true)
					.build();
		} else {
			options = null;
		}

		// For below setup of Runnable, ToDoubleFunction, ... see also:
		// https://stackoverflow.com/a/40153253

		// wrap function to count evaluations
		// and keep track of lowest value encountered so far
		ToDoubleFunction<Matrix> funWrapped = (Matrix x) -> {
			numFunctionEvals++;

			// Send a copy because the user may overwrite it.
            // Overwriting results in undefined behavior because
            // fun(this.x) will change this.x, with the two no longer linked.
			double fx = fun.applyAsDouble(Matrix.Factory.copyFromMatrix(x), args);

			// keep track of lowest value encountered so far
			if (fx < lowestF) {
				lowestF = fx;
				lowestX = x;
			}

			return fx;
		};

		updateFunImpl = () -> {
			this.f = funWrapped.applyAsDouble(this.x);
		};
		updateFun();

		// Gradient evaluation
		final Runnable updateGrad;
		final Function<Matrix, Matrix> gradWrapped;
		if (grad != null) {
			gradWrapped = (Matrix x) -> {
				numGradientEvals++;
				return grad.apply(Matrix.Factory.copyFromMatrix(x), args);
			};
			updateGrad = () -> {
				this.g = gradWrapped.apply(this.x);
			};
		} else if (gradFD != null) {
			gradWrapped = null;
			updateGrad = () -> {
				this.updateFun();
				numGradientEvals++;
				this.g = NumDiff.approxDerivative(funWrapped, x, f, options);
			};
		} else {
			throw new RuntimeException("need either grad or gradFD");
		}
		this.updateGradImpl = updateGrad;
		updateGrad();

		// Hessian Evaluation
		final Runnable updateHess;
		if (hess != null) {
			H = hess.apply(Matrix.Factory.copyFromMatrix(x0), args);
			updatedH = true;
			numHessianEvals++;

			// TODO: sparse Hessian

			// TODO: LinearOperator as Hessian

			java.util.function.Function<Matrix, Matrix> hessWrapped = (Matrix x) -> {
				numHessianEvals++;
				return hess.apply(Matrix.Factory.copyFromMatrix(x), args);
			};
			updateHess = () -> {
				this.H = hessWrapped.apply(this.x);
			};
		} else if (hessFD != null) {
			updateHess = () -> {
				updateGrad();
				this.g = NumDiff.approxDerivative(gradWrapped, x, g, options);
			};
			updateHess();
			updatedH = true;
		} else if (hessStrat != null) {
			hStrat = hessStrat;
			hStrat.initialize(n, HessianApproximationType.HESSIAN);
			updatedH = true;
			xPrev = null;
			gPrev = null;
			updateHess = () -> {
				updateGrad();
				hStrat.update(x.minus(xPrev), g.minus(gPrev));
			};
		} else {
			throw new RuntimeException("need either hess, hessFD or hessStrat");
		}
		updateHessImpl = updateHess;

		Consumer<Matrix> updateX;
		if (hessStrat != null) {
			updateX = (Matrix x) -> {
				// need to keep track of position and gradient
                // for HessianUpdateStrategy
				updateGrad(); // This is free if the gradient is up-to-date.
				xPrev = this.x;
				gPrev = this.g;

				// ensure that self.x is a copy of x. Don't store a reference
                // otherwise the memoization doesn't work properly.
				this.x = Matrix.Factory.copyFromMatrix(x);
				updatedF = false;
				updatedG = false;
				updatedH = false;
				updateHess();
			};
		} else {
			updateX = (Matrix x) -> {
				// ensure that self.x is a copy of x. Don't store a reference
                // otherwise the memoization doesn't work properly.
				this.x = Matrix.Factory.copyFromMatrix(x);
				updatedF = false;
				updatedG = false;
				updatedH = false;
			};
		}
		updateXImpl = updateX;
	}

	private void updateFun() {
		if (!updatedF) {
			updateFunImpl.run();
			updatedF = true;
		}
	}

	private void updateGrad() {
		if (!updatedG) {
			updateGradImpl.run();
			updatedG = true;
		}
	}

	private void updateHess() {
		if (!updatedH) {
			updateHessImpl.run();
			updatedH = true;
		}
	}




	public double fun(Matrix x) {
		if (!this.x.equalsContent(x)) {
			updateXImpl.accept(x);
		}
		updateFun();
		return this.f;
	}

	public Matrix grad(Matrix x) {
		if (!this.x.equalsContent(x)) {
			updateXImpl.accept(x);
		}
		updateGrad();
		return this.g;
	}

	public Matrix hess(Matrix x) {
		if (!this.x.equalsContent(x)) {
			updateXImpl.accept(x);
		}
		updateHess();
		return this.H;
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

	public double f() {
		return f;
	}

	/**
	 * DANGER ZONE: write access to internal state!
	 * @return
	 */
	public Matrix g() {
		return g;
	}

	/**
	 * DANGER ZONE: write access to internal state!
	 * @return
	 */
	public Matrix H() {
		return H;
	}

	public double lowestF() {
		return lowestF;
	}

	public Matrix lowestX() {
		return Matrix.Factory.copyFromMatrix(lowestX);
	}
}
