package de.labathome.trustconstr.records;

import de.labathome.trustconstr.enums.FiniteDifferenceMethod;
import de.labathome.trustconstr.matrix.Matrix;

/**
 * Configuration for {@link de.labathome.trustconstr.NumDiff#approxDerivative}.
 * Build via {@link FiniteDifferenceOptionsFactory} (chained setters then
 * {@link FiniteDifferenceOptionsFactory#build()}).
 */
public final class FiniteDifferenceOptions {

	/** Builder for {@link FiniteDifferenceOptions}. */
	public static class FiniteDifferenceOptionsFactory {

		private FiniteDifferenceMethod method;
		private Matrix relStep;
		private Matrix absStep;
		private FiniteDifferenceBounds bounds;
		private boolean hasBounds;
		private boolean asLinearOperator;
		private Sparsity sparsity;

		/** Default-construct a fresh factory (THREE_POINT, unbounded). */
		public FiniteDifferenceOptionsFactory() {
			method = FiniteDifferenceMethod.THREE_POINT;
			hasBounds = false;
		}

		/**
		 * @param method FD scheme to use
		 * @return this factory, for chaining
		 */
		public FiniteDifferenceOptionsFactory method(FiniteDifferenceMethod method) {
			this.method = method;
			return this;
		}

		/**
		 * @param relStep per-variable relative step size; may be {@code null}
		 *                to derive a default from machine epsilon
		 * @return this factory, for chaining
		 */
		public FiniteDifferenceOptionsFactory relStep(Matrix relStep) {
			this.relStep = relStep;
			return this;
		}

		/**
		 * @param absStep per-variable absolute step size; takes precedence over
		 *                {@link #relStep(Matrix)} when set
		 * @return this factory, for chaining
		 */
		public FiniteDifferenceOptionsFactory absStep(Matrix absStep) {
			this.absStep = absStep;
			return this;
		}

		/**
		 * @param bounds box bounds within which FD perturbations must stay
		 * @return this factory, for chaining
		 * @throws RuntimeException if called more than once
		 */
		public FiniteDifferenceOptionsFactory bounds(FiniteDifferenceBounds bounds) {
			if (hasBounds) {
				throw new RuntimeException("bounds have already been specified");
			} else {
				this.bounds = bounds;
				hasBounds = true;
			}
			return this;
		}

		/**
		 * @param asLinearOperator if {@code true}, return the FD Jacobian as a
		 *                         {@link de.labathome.trustconstr.interfaces.LinearOperator}
		 *                         instead of materialising the full matrix
		 * @return this factory, for chaining
		 */
		public FiniteDifferenceOptionsFactory asLinearOperator(boolean asLinearOperator) {
			this.asLinearOperator = asLinearOperator;
			return this;
		}

		/**
		 * @param sparsity Jacobian sparsity hint enabling Curtis-Powell-Reid
		 *                 column grouping (mass-evaluating columns whose
		 *                 sparsity patterns are row-disjoint)
		 * @return this factory, for chaining
		 */
		public FiniteDifferenceOptionsFactory sparsity(Sparsity sparsity) {
			this.sparsity = sparsity;
			return this;
		}

		/** @return a fresh {@link FiniteDifferenceOptions} with the configured fields */
		public FiniteDifferenceOptions build() {

			if (!hasBounds) {
				bounds = FiniteDifferenceBounds.unbounded(1);
			}

			return new FiniteDifferenceOptions(method, relStep, absStep, bounds, asLinearOperator, sparsity);
		}
	};

	/** Default factory; build via {@link FiniteDifferenceOptionsFactory#build()}. */
	public static final FiniteDifferenceOptionsFactory FACTORY;
	static {
		FACTORY = new FiniteDifferenceOptionsFactory();
	}

	private FiniteDifferenceMethod method;
	private Matrix relStep;
	private Matrix absStep;
	private FiniteDifferenceBounds bounds;
	private boolean asLinearOperator;
	private Sparsity sparsity;

	private FiniteDifferenceOptions(FiniteDifferenceMethod method, Matrix relStep, Matrix absStep,
			FiniteDifferenceBounds bounds, boolean asLinearOperator, Sparsity sparsity) {
		this.method = method;
		this.relStep = relStep;
		this.absStep = absStep;
		this.bounds = bounds;
		this.asLinearOperator = asLinearOperator;
		this.sparsity = sparsity;
	}

	/** @return the FD scheme */
	public FiniteDifferenceMethod method() {
		return method;
	}

	/** @return per-variable relative step size, or {@code null} */
	public Matrix relStep() {
		return relStep;
	}

	/** @return per-variable absolute step size, or {@code null} */
	public Matrix absStep() {
		return absStep;
	}

	/** @return bounds within which FD perturbations must stay */
	public FiniteDifferenceBounds bounds() {
		return bounds;
	}

	/** @return whether the result should be a {@link
	 *  de.labathome.trustconstr.interfaces.LinearOperator} rather than a
	 *  materialised matrix */
	public boolean asLinearOperator() {
		return asLinearOperator;
	}

	/** @return Jacobian sparsity hint, or {@code null} for dense FD */
	public Sparsity sparsity() {
		return sparsity;
	}
}
