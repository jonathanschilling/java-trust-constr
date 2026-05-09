package de.labathome.trustconstr;

import de.labathome.trustconstr.enums.HessianApproximationType;
import de.labathome.trustconstr.interfaces.HessianUpdateStrategy;
import de.labathome.trustconstr.matrix.DenseMatrix;
import de.labathome.trustconstr.matrix.Matrix;

/**
 * Hessian update strategy with full dimensional internal representation.
 */
public abstract class FullHessianUpdateStrategy implements HessianUpdateStrategy {

	/** Shared base factory for {@link BFGS.BFGSFactory} and {@link SR1.SR1Factory}. */
	public static class FullHessianUpdateStrategyFactory {

		/** Whether the initial Hessian scale is auto-derived from the first step. */
		protected boolean initialScaleAuto;
		/** User-specified initial scale, used when {@link #initialScaleAuto} is {@code false}. */
		protected double initialScale;
		private boolean hasInitialScale;

		/** Default-construct: initial scale is auto-derived. */
		protected FullHessianUpdateStrategyFactory() {
			initialScaleAuto = true;
			initialScale = Double.NaN;
			hasInitialScale = false;
		}

		/**
		 * Use the auto-derived initial scale (the default).
		 *
		 * @return this factory, for chaining
		 * @throws RuntimeException if {@link #initalScale(double)} was already called
		 */
		public FullHessianUpdateStrategyFactory initialScaleAuto() {
			if (hasInitialScale) {
				throw new RuntimeException("You can only specify either initScaleAuto or initScale(double).");
			} else {
				this.initialScaleAuto = true;
				this.initialScale = Double.NaN;
				this.hasInitialScale = true;
			}
			return this;
		}

		/**
		 * Pin the initial Hessian scale to a user-supplied value.
		 *
		 * @param initialScale scaling factor for the identity initialisation
		 * @return this factory, for chaining
		 * @throws RuntimeException if {@link #initialScaleAuto()} was already called
		 */
		public FullHessianUpdateStrategyFactory initalScale(double initialScale) {
			if (hasInitialScale) {
				throw new RuntimeException("You can only specify either initialScaleAuto or initalScale(double).");
			} else {
				this.initialScaleAuto = false;
				this.initialScale = initialScale;
				this.hasInitialScale = true;
			}
			return this;
		}

		// no build() method here, since FullHessianUpdateStrategy is abstract
	};

	private final double initialScale;
	private final boolean initScaleAuto;

	/** Current scaling factor applied to the Hessian (or its inverse). */
	protected double scale;

	/** {@code true} until the first {@link #update(Matrix, Matrix)} call has been performed. */
	protected boolean firstIteration;
	/** Whether {@link #B} (HESSIAN) or {@link #H} (INV_HESSIAN) is being approximated. */
	protected HessianApproximationType approxType;

	/** Problem dimension. */
	protected long n;

	/** Hessian approximation (used when {@code approxType == HESSIAN}). */
	protected DenseMatrix B;

	/** Inverse-Hessian approximation (used when {@code approxType == INV_HESSIAN}). */
	protected DenseMatrix H;

	/**
	 * @param initialScaleAuto whether the initial scale is auto-derived
	 * @param initialScale     user-specified initial scale; ignored when
	 *                         {@code initialScaleAuto} is {@code true}
	 */
	protected FullHessianUpdateStrategy(boolean initialScaleAuto, double initialScale) {
		this.initScaleAuto = initialScaleAuto;
		this.initialScale = initialScale;

		firstIteration = false;
		approxType = null;
	}

	@Override
	public void initialize(long n, HessianApproximationType approxType) {
		this.n = n;
		this.approxType = approxType;

		firstIteration = true;

		// Create matrix
		switch (this.approxType) {
		case HESSIAN:
			this.B = DenseMatrix.eye((int) n);
			break;
		case INV_HESSIAN:
			this.H = DenseMatrix.eye((int) n);
			break;
		default:
			throw new RuntimeException("not implemented");
		}
	}

	/**
	 * Heuristic scaling factor for the initial Hessian (or inverse-Hessian)
	 * approximation, based on Nocedal &amp; Wright (2006), p.143 formula (6.20).
	 *
	 * @param deltaX step in {@code x}: {@code x_k - x_{k-1}}
	 * @param deltaG step in the gradient: {@code g_k - g_{k-1}}
	 * @return scaling factor used to multiply the identity initialisation
	 */
	protected double autoScale(Matrix deltaX, Matrix deltaG) {
		double sNorm2 = deltaX.transpose().mtimes(deltaX).doubleValue();
		double yNorm2 = deltaG.transpose().mtimes(deltaG).doubleValue();

		double ys = Math.abs(deltaG.transpose().mtimes(deltaX).doubleValue());

		if (ys == 0.0 || yNorm2 == 0.0 || sNorm2 == 0.0) {
			// fallback to no scaling
			return 1.0;
		}

		switch (approxType) {
		case HESSIAN:
			return yNorm2 / ys;
		case INV_HESSIAN:
			return ys / yNorm2;
		default:
			throw new RuntimeException("not implemented");
		}
	}

	abstract void updateImplementation(Matrix deltaX, Matrix deltaG);

	@Override
	public void update(Matrix deltaX, Matrix deltaG) {

		if (deltaX.normInf() == 0.0) {
			return;
		}

		if (deltaG.normInf() == 0.0) {
			System.out.println("delta_grad == 0.0. Check if the approximated\n" +
					"function is linear. If the function is linear\n" +
					"better results can be obtained by defining the\n" +
					"Hessian as zero instead of using quasi-Newton\n" +
					"approximations.");
			return;
		}

		if (firstIteration) {
			// Get user specific scale
			if (initScaleAuto) {
				scale = autoScale(deltaX, deltaG);
			} else {
				scale = initialScale;
			}

			// Scale initial matrix with ``scale * np.eye(n)``
			switch (approxType) {
			case HESSIAN:
				B = B.times(scale);
				break;
			case INV_HESSIAN:
				H = H.times(scale);
				break;
			default:
				throw new RuntimeException("not implemented");
			}

			firstIteration = false;
		}

		updateImplementation(deltaX, deltaG);
	}

	@Override
	public Matrix apply(Matrix p, Object args) {
		// TODO: use _symv from LAPACK

		switch (approxType) {
		case HESSIAN:
			return B.mtimes(p);
		case INV_HESSIAN:
			return H.mtimes(p);
		default:
			throw new RuntimeException("not implemented");
		}
	}

	@Override
	public Matrix getMatrix() {
		switch (approxType) {
		case HESSIAN:
			return B.copy();
		case INV_HESSIAN:
			return H.copy();
		default:
			throw new RuntimeException("not implemented");
		}
	}
}
