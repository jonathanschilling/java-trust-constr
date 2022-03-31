package org.scipy.optimize.minimize.records;

import org.scipy.optimize.minimize.enums.FiniteDifferenceMethod;
import org.ujmp.core.Matrix;

public final class FiniteDifferenceOptions {

	public static class FiniteDifferenceOptionsFactory {

		private FiniteDifferenceMethod method;
		private Matrix relStep;
		private Matrix absStep;
		private FiniteDifferenceBounds bounds;
		private boolean asLinearOperator;
		private Sparsity sparsity;

		private FiniteDifferenceOptionsFactory() {
			method = FiniteDifferenceMethod.THREE_POINT;
		}

		public FiniteDifferenceOptionsFactory method(FiniteDifferenceMethod method) {
			this.method = method;
			return this;
		}

		public FiniteDifferenceOptionsFactory relStep(Matrix relStep) {
			this.relStep = relStep;
			return this;
		}

		public FiniteDifferenceOptionsFactory absStep(Matrix absStep) {
			this.absStep = absStep;
			return this;
		}

		public FiniteDifferenceOptionsFactory bounds(FiniteDifferenceBounds bounds) {
			this.bounds = bounds;
			return this;
		}

		public FiniteDifferenceOptionsFactory asLinearOperator(boolean asLinearOperator) {
			this.asLinearOperator = asLinearOperator;
			return this;
		}

		public FiniteDifferenceOptionsFactory sparsity(Sparsity sparsity) {
			this.sparsity = sparsity;
			return this;
		}

		public FiniteDifferenceOptions build() {
			return new FiniteDifferenceOptions(method, relStep, absStep, bounds, asLinearOperator, sparsity);
		}
	};

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

	public FiniteDifferenceMethod method() {
		return method;
	}

	public Matrix relStep() {
		return relStep;
	}

	public Matrix absStep() {
		return absStep;
	}

	public FiniteDifferenceBounds bounds() {
		return bounds;
	}

	public boolean asLinearOperator() {
		return asLinearOperator;
	}

	public Sparsity sparsity() {
		return sparsity;
	}
}
