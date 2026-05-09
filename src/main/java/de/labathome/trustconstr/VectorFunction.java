package de.labathome.trustconstr;

import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;

import de.labathome.trustconstr.enums.FiniteDifferenceMethod;
import de.labathome.trustconstr.enums.HessianApproximationType;
import de.labathome.trustconstr.interfaces.HessianUpdateStrategy;
import de.labathome.trustconstr.interfaces.LinearOperator;
import de.labathome.trustconstr.interfaces.VectorFunctionLike;
import de.labathome.trustconstr.records.FiniteDifferenceBounds;
import de.labathome.trustconstr.records.FiniteDifferenceOptions;
import de.labathome.trustconstr.records.Sparsity;
import de.labathome.trustconstr.matrix.DenseMatrix;
import de.labathome.trustconstr.matrix.Matrix;
import de.labathome.trustconstr.matrix.SparseMatrix;

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
public class VectorFunction implements VectorFunctionLike {

	/** Builder for {@link VectorFunction}. */
	public static class VectorFunctionFactory {

		private Function<Matrix, Matrix> fun;
		private Matrix x0;

		private Function<Matrix, Matrix> jac;
		private FiniteDifferenceMethod jacFD;
		private Matrix finiteDiffJacSparsity;
		private boolean hasJac;
		private Optional<Boolean> sparseJacobian;

		private BiFunction<Matrix, Matrix, Matrix> hess;
		private FiniteDifferenceMethod hessFD;
		private HessianUpdateStrategy hessStrat;
		private boolean hasHess;

		private Matrix finiteDiffRelStep;
		private FiniteDifferenceBounds finiteDiffBounds;

		private VectorFunctionFactory() {
			hasJac = false;
			hasHess = false;
			sparseJacobian = Optional.empty();
		}

		/**
		 * Set the vector-valued function to evaluate.
		 *
		 * @param fun function {@code R^n -> R^m}
		 * @return this factory, for chaining
		 */
		public VectorFunctionFactory fun(Function<Matrix, Matrix> fun) {
			this.fun = fun;
			return this;
		}

		/**
		 * Provide the initial point.
		 *
		 * @param x0 starting iterate ({@code n x 1})
		 * @return this factory, for chaining
		 */
		public VectorFunctionFactory x0(Matrix x0) {
			this.x0 = x0;
			return this;
		}


		/**
		 * Provide an analytic Jacobian.
		 *
		 * @param jac analytic Jacobian {@code R^n -> R^{m x n}}
		 * @return this factory, for chaining
		 * @throws RuntimeException if a Jacobian was already specified
		 */
		public VectorFunctionFactory jac(Function<Matrix, Matrix> jac) {
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
		 * Compute the Jacobian by finite differences instead of analytically.
		 *
		 * @param jac FD scheme to use
		 * @return this factory, for chaining
		 * @throws RuntimeException if a Jacobian was already specified
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
		 * Specify a Jacobian sparsity pattern: a matrix with non-zero elements
		 * exactly where non-zero Jacobian entries are expected. Enables
		 * Curtis-Powell-Reid column grouping for the FD path.
		 *
		 * @param finiteDiffJacSparsity sparsity pattern ({@code m x n})
		 * @return this factory, for chaining
		 */
		public VectorFunctionFactory finiteDiffJacSparsity(Matrix finiteDiffJacSparsity) {
			this.finiteDiffJacSparsity = finiteDiffJacSparsity;
			return this;
		}

		/**
		 * Force the Jacobian return type to a sparse representation.
		 *
		 * @return this factory, for chaining
		 */
		public VectorFunctionFactory sparseJacobian() {
			sparseJacobian = Optional.of(true);
			return this;
		}

		/**
		 * Provide an analytic constraint Hessian-of-Lagrangian
		 * {@code (x, v) -> Sum v[i] H_{c_i}(x)}.
		 *
		 * @param hess analytic Hessian-of-Lagrangian
		 * @return this factory, for chaining
		 * @throws RuntimeException if a Hessian was already specified
		 */
		public VectorFunctionFactory hess(BiFunction<Matrix, Matrix, Matrix> hess) {
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
		 * Cannot be combined with an FD Jacobian.
		 *
		 * @param hess FD scheme to use for the Hessian
		 * @return this factory, for chaining
		 * @throws RuntimeException if a Hessian was already specified
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
		 * Approximate the Hessian via a quasi-Newton update strategy
		 * (e.g. {@link BFGS}, {@link SR1}).
		 *
		 * @param hess Hessian-update strategy instance
		 * @return this factory, for chaining
		 * @throws RuntimeException if a Hessian was already specified
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
		 * Per-component relative step size for the FD Jacobian/Hessian. May
		 * be {@code null} to derive a default from machine epsilon.
		 *
		 * @param finiteDiffRelStep relative step size ({@code n x 1}); may be {@code null}
		 * @return this factory, for chaining
		 */
		public VectorFunctionFactory finiteDiffRelStep(Matrix finiteDiffRelStep) {
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
		public VectorFunctionFactory finiteDiffBounds(FiniteDifferenceBounds finiteDiffBounds) {
			this.finiteDiffBounds = finiteDiffBounds;
			return this;
		}

		/**
		 * @return a fully-configured {@link VectorFunction}
		 * @throws RuntimeException if neither analytic nor FD Jacobian/Hessian was set
		 */
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
					jac, jacFD, finiteDiffJacSparsity,
					hess, hessFD, hessStrat,
					finiteDiffRelStep, finiteDiffBounds,
					sparseJacobian);

		}
	};

	/** Default {@link VectorFunctionFactory}. */
	public static final VectorFunctionFactory FACTORY;
	static {
		FACTORY = new VectorFunctionFactory();
	}

	/** current position */
	private Matrix x;
	private Matrix xPrev;

	/** number of parameters */
	private long n;

	/** [m] Lagrange multipliers */
	private Matrix v;

	/** number of function dimensions */
	private long m;

	private int numFunctionEvals;
	private int numJacobianEvals;
	private int numHessianEvals;

	private Matrix f;
	private boolean updatedF;

	private Matrix J;
	private Matrix JPrev;
	private boolean updatedJ;
	private final boolean sparseJacobian;

	private Matrix H;
	private HessianUpdateStrategy hStrat;
	private boolean updatedH;

	private Runnable updateFunImpl;
	private Runnable updateJacImpl;
	private Runnable updateHessImpl;
	private Consumer<Matrix> updateXImpl;

	private VectorFunction(Function<Matrix, Matrix> fun, Matrix x0,
			Function<Matrix, Matrix> jac, FiniteDifferenceMethod jacFD, Matrix finiteDiffJacSparsity,
			BiFunction<Matrix, Matrix, Matrix> hess, FiniteDifferenceMethod hessFD, HessianUpdateStrategy hessStrat,
			Matrix finiteDiffRelStep, FiniteDifferenceBounds finiteDiffBounds,
			Optional<Boolean> sparseJacobian) {

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
			final Sparsity jacSparsity;
			if (finiteDiffJacSparsity != null) {
				int[] sparsityGroups = NumDiff.groupColumns(finiteDiffJacSparsity);
				jacSparsity = new Sparsity(finiteDiffJacSparsity, sparsityGroups);
			} else {
				jacSparsity = null;
			}
			options = FiniteDifferenceOptions.FACTORY
					.method(jacFD)
					.relStep(finiteDiffRelStep)
					.sparsity(jacSparsity)
					.build();
		} else if (hessFD != null) {
			options = FiniteDifferenceOptions.FACTORY
					.method(hessFD)
					.relStep(finiteDiffRelStep)
					.asLinearOperator(true)
					.build();
		} else {
			options = null;
		}

		Function<Matrix, Matrix> funWrapped = (Matrix x) -> {
			numFunctionEvals++;
			return fun.apply(x);
		};

		Runnable updateFun = () -> {
			f = funWrapped.apply(x);
		};
		updateFunImpl = updateFun;
		updateFun.run();
		//updateFun(); // TODO: why not updateFun();

		v = Matrix.Factory.zeros(f.getSize());
		m = v.getRowCount();

		// Jacobian Evaluation
		final Runnable updateJac;
		final Function<Matrix, Matrix> jacWrapped;
		if (jac != null) {
			J = jac.apply(x);
			updatedJ = true;
			numJacobianEvals++;

			if ( (sparseJacobian.isPresent() && sparseJacobian.get()) ||
					(sparseJacobian.isEmpty() && J.isSparse()) ) {
				jacWrapped = (Matrix x) -> {
					numJacobianEvals++;
					return SparseMatrix.Factory.copyFromMatrix(jac.apply(x));
				};
				this.sparseJacobian = true;
			} else if (J.isSparse()) {
				// sparseJacobian was set to false, but the Jacobian is sparse,
				// so need to convert it to a dense matrix
				jacWrapped = (Matrix x) -> {
					numJacobianEvals++;
					return DenseMatrix.Factory.copyFromMatrix(jac.apply(x));
				};
				J = DenseMatrix.Factory.copyFromMatrix(J);
				this.sparseJacobian = false;
			} else {
				// dense Jacobian
				jacWrapped = (Matrix x) -> {
					numJacobianEvals++;
					return jac.apply(x);
				};
				this.sparseJacobian = false;
			}

			updateJac = () -> {
				J = jacWrapped.apply(x);
			};
		} else if (jacFD != null) {
			J = NumDiff.approxDerivative(funWrapped, x, f, options);
			updatedJ = true;

			if ( (sparseJacobian.isPresent() && sparseJacobian.get()) ||
					(sparseJacobian.isEmpty() && J.isSparse()) ) {
				updateJac = () -> {
					updateFun();
					J = SparseMatrix.Factory.copyFromMatrix(
							NumDiff.approxDerivative(funWrapped, x, f, options));
				};
				J = SparseMatrix.Factory.copyFromMatrix(J);
				this.sparseJacobian = true;
			} else if (J.isSparse()) {
				// sparseJacobian was set to false, but the Jacobian is sparse,
				// so need to convert it to a dense matrix
				updateJac = () -> {
					updateFun();
					J = DenseMatrix.Factory.copyFromMatrix(NumDiff.approxDerivative(funWrapped, x, f, options));
				};
				J = DenseMatrix.Factory.copyFromMatrix(J);
				this.sparseJacobian = false;
			} else {
				// dense Jacobian
				updateJac = () -> {
					updateFun();
					J = DenseMatrix.Factory.copyFromMatrix(NumDiff.approxDerivative(funWrapped, x, f, options));
				};
				J = DenseMatrix.Factory.copyFromMatrix(J);
				this.sparseJacobian = false;
			}

			jacWrapped = null;
		} else {
			throw new RuntimeException("No way to obtain the Jacobian was found.");
		}
		updateJacImpl = updateJac;

		// Define Hessian
		final Runnable updateHess;
		if (hess != null) {
			H = hess.apply(x, v);
			updatedH = true;
			numHessianEvals++;

			final BiFunction<Matrix, Matrix, Matrix> hessWrapped;
			if (H.isSparse()) {
				hessWrapped = (Matrix x, Matrix v) -> {
					numHessianEvals++;
					return SparseMatrix.Factory.copyFromMatrix(hess.apply(x, v));
				};
				H = SparseMatrix.Factory.copyFromMatrix(H);
			} else if (H instanceof LinearOperator) {
				hessWrapped = (Matrix x, Matrix v) -> {
					numHessianEvals++;
					// TODO: map hess.apply to invoking dot product in LinearOperator H
					return hess.apply(x, v);
				};
			} else {
				hessWrapped = (Matrix x, Matrix v) -> {
					numHessianEvals++;
					return hess.apply(x, v);
				};
			}

			updateHess = () -> {
				H = hessWrapped.apply(x, v);
			};
		} else if (hessFD != null) {
			BiFunction<Matrix, Object, Matrix> jacDotV = (Matrix x, Object args) -> {
				if (!(args instanceof Matrix)) {
					throw new RuntimeException("jacDotV must be called with Matrix v as second argument.");
				}
				Matrix v = (Matrix) args;
				return jacWrapped.apply(x).transpose().mtimes(v);
			};
			updateHess = () -> {
				updateJac();
				H = NumDiff.approxDerivative(jacDotV, x,
						J.transpose().mtimes(v), options, v);
			};
			updateHess.run();
			updatedH = true;
		} else if (hessStrat != null) {
			hStrat = hessStrat;
			hStrat.initialize(n, HessianApproximationType.HESSIAN);
			updatedH = true;
			xPrev = null;
			JPrev = null;
			updateHess = () -> {
				updateJac();
				// When v is updated before x was updated,
				// then x_prev and J_prev are None and we need this check.
				if (xPrev != null && JPrev != null) {
					Matrix deltaX = x.minus(xPrev);
					Matrix deltaG = (J.transpose().mtimes(v)).minus(JPrev.transpose().mtimes(v));
					hStrat.update(deltaX, deltaG);
				}
			};
		}
		else {
			throw new RuntimeException("No way to obtain the Hessian was found:.");
		}
		updateHessImpl = updateHess;

		Consumer<Matrix> updateX;
		if (hessStrat != null) {
			updateX = (Matrix x) -> {
				// need to keep track of position and Jacobian for HessianUpdateStrategy
				updateJac(); // This is free if the Jacobian is up-to-date.
				xPrev = this.x;
				JPrev = this.J;

				// ensure that self.x is a copy of x. Don't store a reference
                // otherwise the memoization doesn't work properly.
				this.x = Matrix.Factory.copyFromMatrix(x);
				updatedF = false;
				updatedJ = false;
				updatedH = false;
				updateHess();
			};
		} else {
			updateX = (Matrix x) -> {
				// ensure that self.x is a copy of x. Don't store a reference
                // otherwise the memoization doesn't work properly.
				this.x = Matrix.Factory.copyFromMatrix(x);
				updatedF = false;
				updatedJ = false;
				updatedH = false;
			};
		}
		updateXImpl = updateX;
	}

	private void updateV(Matrix v) {
		if (!this.v.equalsContent(v)) {
			 this.v = v;
			 updatedH = false;
		}
	}

	private void updateX(Matrix x) {
		if (!this.x.equalsContent(x)) {
			 updateXImpl.accept(x);
		}
	}

	private void updateFun() {
		if (!updatedF) {
			updateFunImpl.run();
			updatedF = true;
		}
	}

	private void updateJac() {
		if (!updatedJ) {
			updateJacImpl.run();
			updatedJ = true;
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
	 * @return cached function value {@code f(x)} ({@code m x 1})
	 */
	public Matrix fun(Matrix x) {
		updateX(x);
		updateFun();
		return this.f;
	}

	/**
	 * @param x current iterate ({@code n x 1})
	 * @return cached Jacobian {@code J(x)} ({@code m x n})
	 */
	public Matrix jac(Matrix x) {
		updateX(x);
		updateJac();
		return this.J;
	}

	/**
	 * @param x current iterate ({@code n x 1})
	 * @param v Lagrange multipliers ({@code m x 1})
	 * @return cached Hessian-of-Lagrangian {@code H(x, v)} ({@code n x n})
	 */
	public Matrix hess(Matrix x, Matrix v) {
		// v should be updated before x.
		updateV(v);
		updateX(x);
		updateHess();
		return this.H;
	}

	/** @return total number of {@code fun} evaluations performed so far */
	public int numFunctionEvals() {
		return numFunctionEvals;
	}

	/** @return total number of Jacobian evaluations performed so far */
	public int numJacobianEvals() {
		return numJacobianEvals;
	}

	/** @return total number of Hessian evaluations performed so far */
	public int numHessianEvals() {
		return numHessianEvals;
	}

	/** @return {@code true} if the Jacobian return type is sparse */
	public boolean sparseJacobian() {
		return sparseJacobian;
	}

	/** @return the most recently computed {@code f(x)} ({@code m x 1}), or {@code null} if uncached */
	public Matrix f() {
		return f;
	}

	/** @return the most recently computed Jacobian ({@code m x n}), or {@code null} if uncached */
	public Matrix J() {
		return J;
	}

	/** @return the most recently computed Hessian ({@code n x n}), or {@code null} if uncached */
	public Matrix H() {
		return H;
	}

	/** @return the most recently set Lagrange multipliers ({@code m x 1}), or {@code null} */
	public Matrix v() {
		return v;
	}

	/** @return number of function dimensions ({@code m} in {@code f : R^n -> R^m}) */
	@Override
	public long m() {
		return m;
	}

	/** @return number of input dimensions ({@code n} in {@code f : R^n -> R^m}) */
	@Override
	public long n() {
		return n;
	}
}
