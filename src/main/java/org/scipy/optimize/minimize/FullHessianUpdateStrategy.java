package org.scipy.optimize.minimize;

import org.scipy.optimize.minimize.enums.HessianApproximationType;
import org.scipy.optimize.minimize.interfaces.HessianUpdateStrategy;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;

/**
 * Hessian update strategy with full dimensional internal representation.
 */
public abstract class FullHessianUpdateStrategy implements HessianUpdateStrategy {

	public static class FullHessianUpdateStrategyFactory {

		protected boolean initialScaleAuto;
		protected double initialScale;
		private boolean hasInitialScale;

		protected FullHessianUpdateStrategyFactory() {
			initialScaleAuto = true;
			initialScale = Double.NaN;
			hasInitialScale = false;
		}

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
	};

	private final double initialScale;
	private final boolean initScaleAuto;

	protected double scale;

	protected boolean firstIteration;
	protected HessianApproximationType approxType;

	/** problem dimension */
	protected long n;

	/** Hessian */
	protected Matrix B;

	/** inverse Hessian */
	protected Matrix H;

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
			this.B = Matrix.Factory.eye(n, n);
			break;
		case INV_HESSIAN:
			this.H = Matrix.Factory.eye(n, n);
			break;
		default:
			throw new RuntimeException("not implemented");
		}
	}

	/**
	 * Heuristic to scale matrix at first iteration.
	 * Described in Nocedal and Wright "Numerical Optimization"
	 * p.143 formula (6.20).
	 *
	 * @param deltaX
	 * @param deltaG
	 */
	protected double autoScale(Matrix deltaX, Matrix deltaG) {
		double sNorm2 = deltaX.mtimes(deltaX).doubleValue();
		double yNorm2 = deltaG.mtimes(deltaG).doubleValue();

		double ys = Math.abs(deltaG.mtimes(deltaX).doubleValue());

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
	public Matrix dot(Matrix p) {

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
			return DenseMatrix.Factory.copyFromMatrix(B);
		case INV_HESSIAN:
			return DenseMatrix.Factory.copyFromMatrix(H);
		default:
			throw new RuntimeException("not implemented");
		}
	}

	@Override
	public Matrix hess(Matrix x, Object args) {
		throw new RuntimeException("not implemented yet");
	}
}
