/*
 * Copyright 2026 Jonathan Schilling
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package de.labathome.trustconstr;

import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.ToDoubleBiFunction;
import java.util.function.ToDoubleFunction;

import de.labathome.trustconstr.enums.FiniteDifferenceMethod;
import de.labathome.trustconstr.enums.HessianApproximationType;
import de.labathome.trustconstr.interfaces.HessianUpdateStrategy;
import de.labathome.trustconstr.records.FiniteDifferenceBounds;
import de.labathome.trustconstr.records.FiniteDifferenceOptions;
import de.labathome.trustconstr.matrix.Matrix;

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

	/** Builder for {@link ScalarFunction}. */
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

		/** Default-construct an empty factory. */
		public ScalarFunctionFactory() {
			hasGrad = false;
			hasHess = false;
		}

		/**
		 * Set the objective function to optimize.
		 *
		 * @param fun objective {@code (x, args) -> f(x)}
		 * @return this factory, for chaining
		 */
		public ScalarFunctionFactory fun(ToDoubleBiFunction<Matrix, Object> fun) {
			this.fun = fun;
			return this;
		}

		/**
		 * Provide the initial point for evaluating {@code fun}.
		 *
		 * @param x0 starting iterate ({@code n x 1})
		 * @return this factory, for chaining
		 */
		public ScalarFunctionFactory x0(Matrix x0) {
			this.x0 = x0;
			return this;
		}

		/**
		 * Provide any additional fixed parameters needed to specify the
		 * scalar function (the {@code args} argument forwarded to
		 * {@code fun}/{@code grad}/{@code hess}).
		 *
		 * @param args fixed parameters (may be {@code null})
		 * @return this factory, for chaining
		 */
		public ScalarFunctionFactory args(Object args) {
			this.args = args;
			return this;
		}

		/**
		 * Provide an analytic gradient {@code grad(x, args) -> gradf(x)}.
		 * Mutually exclusive with {@link #grad(FiniteDifferenceMethod)}.
		 *
		 * @param grad analytic gradient
		 * @return this factory, for chaining
		 * @throws RuntimeException if a gradient (analytic or FD) was already specified
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
		 * Compute the gradient by finite differences instead of analytically.
		 * Mutually exclusive with {@link #grad(BiFunction)}.
		 *
		 * @param grad FD scheme to use
		 * @return this factory, for chaining
		 * @throws RuntimeException if a gradient was already specified
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
		 * Provide an analytic Hessian {@code hess(x, args) -> grad^2f(x)}.
		 * Mutually exclusive with the FD and quasi-Newton overloads.
		 *
		 * @param hess analytic Hessian evaluator
		 * @return this factory, for chaining
		 * @throws RuntimeException if a Hessian was already specified
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
		 * Compute the Hessian by finite differences instead of analytically.
		 * Note: cannot be combined with an FD gradient.
		 *
		 * @param hess FD scheme to use for the Hessian
		 * @return this factory, for chaining
		 * @throws RuntimeException if a Hessian was already specified
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
		 * Approximate the Hessian via a quasi-Newton update strategy
		 * (e.g. {@link BFGS}, {@link SR1}).
		 *
		 * @param hess Hessian-update strategy instance
		 * @return this factory, for chaining
		 * @throws RuntimeException if a Hessian was already specified
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
		 * Per-component relative step size for the FD gradient/Hessian. The
		 * absolute step is {@code h = relStep * sign(x0) * max(1, |x0|)},
		 * clamped to {@link #finiteDiffBounds(FiniteDifferenceBounds)}. May be
		 * {@code null} to derive a default from machine epsilon.
		 *
		 * @param finiteDiffRelStep relative step size ({@code n x 1}); may be {@code null}
		 * @return this factory, for chaining
		 */
		public ScalarFunctionFactory finiteDiffRelStep(Matrix finiteDiffRelStep) {
			this.finiteDiffRelStep = finiteDiffRelStep;
			return this;
		}

		/**
		 * Bounds within which FD perturbations must stay. Defaults to
		 * unbounded.
		 *
		 * @param finiteDiffBounds box bounds for FD perturbations
		 * @return this factory, for chaining
		 */
		public ScalarFunctionFactory finiteDiffBounds(FiniteDifferenceBounds finiteDiffBounds) {
			this.finiteDiffBounds = finiteDiffBounds;
			return this;
		}

		/**
		 * Absolute step size for FD gradient/Hessian, taking precedence over
		 * the relative step. May be {@code null} (use relative step instead).
		 *
		 * @param epsilon absolute step size ({@code n x 1}); may be {@code null}
		 * @return this factory, for chaining
		 */
		public ScalarFunctionFactory epsilon(Matrix epsilon) {
			this.epsilon = epsilon;
			return this;
		}

		/**
		 * @return a fully-configured {@link ScalarFunction} ready for use
		 * @throws RuntimeException if neither analytic nor FD gradient/Hessian was set
		 */
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

	/** Default {@link ScalarFunctionFactory}; create one and call {@link ScalarFunctionFactory#build()}. */
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
		// Build a fresh FiniteDifferenceOptionsFactory per ScalarFunction --
		// the static FiniteDifferenceOptions.FACTORY singleton retains state
		// across calls (".bounds() throws 'bounds have already been specified'
		// on the second call"); the same hazard pattern is documented in
		// CLAUDE.md.
		if (gradFD != null) {
			options = new FiniteDifferenceOptions.FiniteDifferenceOptionsFactory()
					.method(gradFD)
					.relStep(finiteDiffRelStep)
					.absStep(epsilon)
					.bounds(finiteDiffBounds)
					.build();
		} else if (hessFD != null) {
			options = new FiniteDifferenceOptions.FiniteDifferenceOptionsFactory()
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

			Function<Matrix, Matrix> hessWrapped = (Matrix x) -> {
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




	/**
	 * @param x current iterate ({@code n x 1})
	 * @return cached objective value {@code f(x)}
	 */
	public double fun(Matrix x) {
		if (!this.x.equalsContent(x)) {
			updateXImpl.accept(x);
		}
		updateFun();
		return this.f;
	}

	/**
	 * @param x current iterate ({@code n x 1})
	 * @return cached gradient {@code gradf(x)} ({@code n x 1})
	 */
	public Matrix grad(Matrix x) {
		if (!this.x.equalsContent(x)) {
			updateXImpl.accept(x);
		}
		updateGrad();
		return this.g;
	}

	/**
	 * @param x current iterate ({@code n x 1})
	 * @return cached Hessian {@code grad^2f(x)} ({@code n x n})
	 */
	public Matrix hess(Matrix x) {
		if (!this.x.equalsContent(x)) {
			updateXImpl.accept(x);
		}
		updateHess();
		return this.H;
	}

	/** @return total number of {@code fun} evaluations performed so far */
	public int numFunctionEvals() {
		return numFunctionEvals;
	}

	/** @return total number of gradient evaluations performed so far */
	public int numGradientEvals() {
		return numGradientEvals;
	}

	/** @return total number of Hessian evaluations performed so far */
	public int numHessianEvals() {
		return numHessianEvals;
	}

	/** @return the most recently cached {@code f(x)} */
	public double f() {
		return f;
	}

	/**
	 * DANGER ZONE: returns the live internal gradient buffer; callers must
	 * not mutate it.
	 *
	 * @return the most recently cached gradient ({@code n x 1})
	 */
	public Matrix g() {
		return g;
	}

	/**
	 * DANGER ZONE: returns the live internal Hessian buffer; callers must
	 * not mutate it.
	 *
	 * @return the most recently cached Hessian ({@code n x n})
	 */
	public Matrix H() {
		return H;
	}

	/** @return the lowest objective value seen across all evaluations */
	public double lowestF() {
		return lowestF;
	}

	/** @return a defensive copy of the iterate that produced {@link #lowestF()} */
	public Matrix lowestX() {
		return Matrix.Factory.copyFromMatrix(lowestX);
	}
}
